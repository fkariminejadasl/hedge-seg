import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from datasets import Dataset
from datasets import Image as HFImage
from datasets import load_from_disk
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from torch.nn.functional import softmax
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoModelForObjectDetection,
    DetrImageProcessor,
    Trainer,
    TrainingArguments,
    pipeline,
)

"""
It is mainly based on https://huggingface.co/learn/cookbook/en/fine_tuning_detr_custom_dataset. 
I made some adjustments to fit my dataset and needs, and I also fixed a few bugs.
"""


def convert_xyxy_to_xywh(xmin, ymin, xmax, ymax):
    # COCO bbox is [x, y, width, height]
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    return [float(xmin), float(ymin), float(w), float(h)]


def make_coco(images_dir, labels_dir, output_json="instances_train.json"):
    coco = {
        "info": {"description": "Converted dataset", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
    }

    ann_id = 1
    img_id = 1

    # Collect common image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Iterate images
    for img_path in sorted(
        [p for p in images_dir.glob("pos_*") if p.suffix.lower() in exts]
    ):
        label_path = labels_dir / (img_path.stem + ".json")
        if not label_path.exists():
            print(f"Skipping (no label): {img_path.name}")
            continue

        # Read image size
        width, height = 256, 256
        # with Image.open(img_path) as im:
        #     width, height = im.size

        # Read label JSON
        with open(label_path, "r") as f:
            lab = json.load(f)

        boxes = lab.get("bboxes_px_per_polyline", [])
        if not isinstance(boxes, list):
            print(f"Skipping (bad format): {label_path.name}")
            continue

        # Add image record
        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,  # relative to images folder
                "width": int(width),
                "height": int(height),
            }
        )

        # Add annotations
        for b in boxes:
            if not (isinstance(b, list) and len(b) == 4):
                continue
            xmin, ymin, xmax, ymax = b

            # Clip to image bounds (optional but helpful)
            xmin = max(0, min(xmin, width))
            xmax = max(0, min(xmax, width))
            ymin = max(0, min(ymin, height))
            ymax = max(0, min(ymax, height))

            bbox_xywh = convert_xyxy_to_xywh(xmin, ymin, xmax, ymax)
            area = bbox_xywh[2] * bbox_xywh[3]
            if area < 0:
                continue
            if area == 0:
                bbox_xywh[2] = max(1.0, bbox_xywh[2])
                bbox_xywh[3] = max(1.0, bbox_xywh[3])
                area = bbox_xywh[2] * bbox_xywh[3]

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "area": float(area),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved COCO to: {output_json}")
    print(f"Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")


def to_hf_detr(images_dir, labels_dir, img_glob="pos_*", label_suffix=".json"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    rows = []
    bbox_id = 1
    image_id = 1

    for img_path in sorted(
        [p for p in images_dir.glob(img_glob) if p.suffix.lower() in exts]
    ):
        lab_path = labels_dir / (img_path.stem + label_suffix)
        if not lab_path.exists():
            continue

        lab = json.load(open(lab_path, "r"))
        boxes = lab.get("bboxes_px_per_polyline", [])
        if not isinstance(boxes, list):
            continue

        # keep minimal: fixed size (swap for PIL if needed)
        width, height = 256, 256

        bboxes, bbox_ids, cats, areas = [], [], [], []
        for b in boxes:
            if not (isinstance(b, list) and len(b) == 4):
                continue
            x1, y1, x2, y2 = map(float, b)

            # clip
            x1 = max(0.0, min(x1, width))
            x2 = max(0.0, min(x2, width))
            y1 = max(0.0, min(y1, height))
            y2 = max(0.0, min(y2, height))

            if x1 > x2 or y1 > y2:
                print(f"Image with invalid bbox: {img_path}")
            if x2 == x1:
                x2 = min(width, x1 + 1.0)
            if y2 == y1:
                y2 = min(height, y1 + 1.0)

            w = x2 - x1
            h = y2 - y1
            area = w * h

            bboxes.append([x1, y1, x2, y2])  # xyxy
            bbox_ids.append(bbox_id)
            bbox_id += 1
            cats.append(0)  # single class -> 0
            areas.append(area)

        rows.append(
            {
                "image_id": image_id,
                "image": str(img_path),
                "width": width,
                "height": height,
                "objects": {
                    "bbox": bboxes,
                    "bbox_id": bbox_ids,
                    "category": cats,
                    "area": areas,
                },
            }
        )
        image_id += 1

    return Dataset.from_list(rows).cast_column("image", HFImage())


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "iscrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def convert_voc_to_coco(bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]


def transform_aug_ann(examples, transform):
    image_ids = examples["image_id"]
    images, bboxes, areas, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(
            image=image,
            bboxes=objects["bbox"],
            category=objects["category"],
            bbox_id=objects["bbox_id"],
        )

        # area.append(objects["area"])
        images.append(out["image"])

        # Convert to COCO format: coco xywh from voc xyxy
        converted_bboxes = [convert_voc_to_coco(b) for b in out["bboxes"]]
        bboxes.append(converted_bboxes)
        # recompute areas aligned with converted_bboxes
        areas.append([bb[2] * bb[3] for bb in converted_bboxes])

        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, areas, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def transform_train(examples):
    return transform_aug_ann(examples, transform=train_transform)


def transform_val(examples):
    return transform_aug_ann(examples, transform=val_transform)


def draw_augmented_image_from_idx(dataset, idx, transform=None):
    sample = dataset[idx]
    image = sample["image"]
    annotations = sample["objects"]

    # Convert image to RGB and NumPy array
    image = np.array(image.convert("RGB"))[:, :, ::-1]

    if transform:
        augmented = transform(
            image=image,
            bboxes=annotations["bbox"],
            category=annotations["category"],
            bbox_id=annotations["bbox_id"],
        )
        image = augmented["image"]
        annotations["bbox"] = augmented["bboxes"]
        annotations["category"] = augmented["category"]
        annotations["bbox_id"] = augmented["bbox_id"]

    image = Image.fromarray(image[:, :, ::-1])  # Convert back to PIL Image
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for i in range(len(annotations["bbox_id"])):
        box = annotations["bbox"][i]
        x1, y1, x2, y2 = tuple(box)

        # Normalize coordinates if necessary
        if max(box) <= 1.0:
            x1, y1 = int(x1 * width), int(y1 * height)
            x2, y2 = int(x2 * width), int(y2 * height)
        else:
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1), id2label[annotations["category"][i]], fill="green")

    return image


def plot_augmented_images(dataset, indices, transform=None):
    """
    Plot images and their annotations with optional augmentation.
    """
    num_rows = len(indices) // 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for i, idx in enumerate(indices):
        row = i // num_cols
        col = i % num_cols

        # Draw augmented image
        image = draw_augmented_image_from_idx(dataset, idx, transform=transform)

        # Display image on the corresponding subplot
        axes[row, col].imshow(image)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def denormalize_boxes(boxes, width, height):
    boxes = boxes.clone()
    boxes[:, 0] *= width  # xmin
    boxes[:, 1] *= height  # ymin
    boxes[:, 2] *= width  # xmax
    boxes[:, 3] *= height  # ymax
    return boxes


batch_metrics = []


def compute_metrics(eval_pred, compute_result):
    global batch_metrics

    (
        loss_dict,
        scores,
        pred_boxes,
        last_hidden_state,
        encoder_last_hidden_state,
    ), labels = eval_pred

    image_sizes = []
    target = []
    for label in labels:

        image_sizes.append(label["orig_size"])
        width, height = label["orig_size"]
        denormalized_boxes = denormalize_boxes(label["boxes"], width, height)
        target.append(
            {
                "boxes": denormalized_boxes,
                "labels": label["class_labels"],
            }
        )
    predictions = []
    for score, box, target_sizes in zip(scores, pred_boxes, image_sizes):
        # Extract the bounding boxes, labels, and scores from the model's output
        pred_scores = score[:, :-1]  # Exclude the no-object class
        pred_scores = softmax(pred_scores, dim=-1)
        width, height = target_sizes
        pred_boxes = denormalize_boxes(box, width, height)
        pred_labels = torch.argmax(pred_scores, dim=-1)

        # Get the scores corresponding to the predicted labels
        pred_scores_for_labels = torch.gather(
            pred_scores, 1, pred_labels.unsqueeze(-1)
        ).squeeze(-1)
        predictions.append(
            {
                "boxes": pred_boxes,
                "scores": pred_scores_for_labels,
                "labels": pred_labels,
            }
        )

    metric = MeanAveragePrecision(box_format="xywh", class_metrics=True)

    if not compute_result:
        # Accumulate batch-level metrics
        batch_metrics.append({"preds": predictions, "target": target})
        return {}
    else:
        # Compute final aggregated metrics
        # Aggregate batch-level metrics (this should be done based on your metric library's needs)
        all_preds = []
        all_targets = []
        for batch in batch_metrics:
            all_preds.extend(batch["preds"])
            all_targets.extend(batch["target"])

        # Update metric with all accumulated predictions and targets
        metric.update(preds=all_preds, target=all_targets)
        metrics = metric.compute()

        # Convert and format metrics as needed
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")

        # make sure they are 1D: in case of single class, it is 0D tensors
        if isinstance(classes, torch.Tensor) and classes.ndim == 0:
            classes = classes.unsqueeze(0)
        if isinstance(map_per_class, torch.Tensor) and map_per_class.ndim == 0:
            map_per_class = map_per_class.unsqueeze(0)
        if isinstance(mar_100_per_class, torch.Tensor) and mar_100_per_class.ndim == 0:
            mar_100_per_class = mar_100_per_class.unsqueeze(0)

        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class
        ):
            class_name = (
                id2label[class_id.item()] if id2label is not None else class_id.item()
            )
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        # Round metrics for cleaner output
        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        # Clear batch metrics for next evaluation
        batch_metrics = []

        return metrics


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }

    # pixel_values = torch.stack([item["pixel_values"] for item in batch])  # [B,3,500,500]
    # labels = [item["labels"] for item in batch]
    # pixel_mask = torch.ones(
    #     (pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]),
    #     dtype=torch.long
    # )

    # return {
    #     "pixel_values": pixel_values, # encoding["pixel_values"]
    #     "pixel_mask": pixel_mask, # encoding["pixel_mask"]
    #     "labels": labels,
    # }


def plot_results(image, results, threshold=0.6):
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for result in results:
        score = result["score"]
        label = result["label"]
        box = list(result["box"].values())

        if score > threshold:
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1 + 5, y1 - 10), label, fill="white")
            draw.text(
                (x1 + 5, y1 + 10),
                f"{score:.2f}",
                fill="green" if score > 0.7 else "red",
            )
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    return image


# # Convert to COCO format
# image_dir = Path("/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/images")
# labels_dir = Path("/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/labels_processed")
# output_json = Path("/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/instances_train.json")
# make_coco(image_dir, labels_dir, output_json)


# # Convert to Hugging Face Datasets format
# ds = to_hf_detr(
#     images_dir="/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/images",
#     labels_dir="/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/labels_processed",
# )
# # print(ds[0]["objects"])
# ds.save_to_disk("/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/detr_dataset")


output_dir = Path("/home/fatemeh/Downloads/hedg/results/training/detr-hf")
# Load the dataset
train_dataset = load_from_disk(
    "/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/detr_dataset"
)
test_dataset = load_from_disk(
    "/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/detr_dataset"
)
checkpoint = "facebook/detr-resnet-50-dc5"
id2label = {0: "object"}
label2id = {"object": 0}

# Augmentation
size = 255
train_transform = A.Compose(
    [
        A.LongestMaxSize(size),
        A.PadIfNeeded(size, size, border_mode=0, value=(0, 0, 0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category", "bbox_id"]),
)

val_transform = A.Compose(
    [
        A.LongestMaxSize(size),
        A.PadIfNeeded(size, size, border_mode=0, value=(0, 0, 0)),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category", "bbox_id"]),
)


train_dataset_transformed = train_dataset.with_transform(transform_train)
test_dataset_transformed = test_dataset.with_transform(transform_val)

# # plot_augmented
# plot_augmented_images(train_dataset, range(6), transform=train_transform)
# # plot_augmented_images(train_dataset, range(6))

image_processor = DetrImageProcessor.from_pretrained(checkpoint)
# image_processor = AutoImageProcessor.from_pretrained(checkpoint) # issue with .pad method
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # reinit head
)
# config = AutoConfig.from_pretrained(
#     checkpoint,
#     num_labels=len(id2label),
#     id2label=id2label,
#     label2id=label2id,
# )
# model = AutoModelForObjectDetection.from_pretrained(
#     checkpoint,
#     config=config,
#     ignore_mismatched_sizes=True,  # reinit head
# )

# Freeze backbone
for name, p in model.named_parameters():
    if "backbone" in name:
        p.requires_grad = False
    else:
        p.requires_grad = True

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # 4,
    per_device_eval_batch_size=2,  # 4,
    max_steps=10,  # 10000,
    fp16=True,
    save_steps=10,
    logging_steps=1,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    eval_steps=1,  # 50,
    eval_strategy="steps",
    report_to="wandb",
    push_to_hub=False,
    batch_eval_metrics=True,
)

import wandb

wandb.init(project=output_dir.stem, name=output_dir.stem, config=training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset_transformed,
    eval_dataset=test_dataset_transformed,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
# image_processor.save_pretrained(training_args.output_dir)


# Test the trained model on a sample image
image = Image.open(
    "/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/images/pos_000000.png"
).convert("RGB")
obj_detector = pipeline("object-detection", model=output_dir)
results = obj_detector(image)
print(results)
plot_results(image, results, threshold=0.6)
metrics = trainer.evaluate(test_dataset_transformed)
print(metrics)
