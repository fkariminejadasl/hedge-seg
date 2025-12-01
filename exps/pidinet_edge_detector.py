"""
I saw in CoVT (Chain of Visual Thought) paper that they used PiDiNet edge detector.
CoVT paper: https://arxiv.org/pdf/2511.19418
PiDiNet: https://arxiv.org/pdf/2108.07009

requires: pip install onnxruntime pillow
"""

import torch
import numpy as np
from PIL import Image

# git clone https://github.com/hellozhuo/pidinet.git
# cd pidinet
# or
import sys
sys.path.append("/home/fatemeh/dev/pidinet")
from models import pidinet_converted
from models.convert_pidinet import convert_pidinet

# https://huggingface.co/bdck/PiDiNet_ONNX
# Download the PiDiNet model checkpoint from https://github.com/hellozhuo/pidinet
CKPT_PATH = "/home/fatemeh/Downloads/table5_pidinet.pth"
IMAGE_PATH = "/home/fatemeh/dev/veg-outline/docs/knepp_95_height_u16.jpg"
OUTPUT_PATH = "/home/fatemeh/dev/veg-outline/edges.png"


# Same normalization as ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

def load_model(ckpt_path):
    # Build the converted PiDiNet model (uses plain convs)
    class Args:
        config = "carv4"
        sa = True
        dil = True

    model = pidinet_converted(Args)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}

    state = convert_pidinet(state, Args.config)
    model.load_state_dict(state)
    model.eval()
    return model

def preprocess(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC
    arr = arr.transpose(2, 0, 1)                      # CHW
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr)[None, ...]         # BCHW
    return tensor, img.size

def postprocess(edge_map, size, out_path):
    edge_map = edge_map.squeeze().detach().cpu().numpy()
    edge_map = np.clip(edge_map, 0.0, 1.0)
    edge_map = (edge_map * 255.0).astype(np.uint8)
    edge_img = Image.fromarray(edge_map)
    edge_img = edge_img.resize(size, Image.BILINEAR)
    edge_img.save(out_path)

def main():
    model = load_model(CKPT_PATH)
    inp, size = preprocess(IMAGE_PATH)

    with torch.no_grad():
        outputs = model(inp)
        # fused has 5 channels, last is fused edge map, others are edge outputs from fine (0) to course (3)
        fused = outputs[-1]  # last output is fused edge map

    postprocess(fused, size, OUTPUT_PATH)
    print(f"Saved edge map to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
