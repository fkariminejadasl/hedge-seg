import json
from pathlib import Path

from hedge_seg.polyline_utils import bbox_from_polyline, resample_polyline_equidistant


def process_label_file(path: Path, out_dir: Path, n_points):
    data = json.loads(path.read_text(encoding="utf-8"))

    polylines = data.get("polylines_px", []) or []

    bboxes = [bbox_from_polyline(line) for line in polylines]
    resampled = [resample_polyline_equidistant(line, n_points) for line in polylines]

    out_data = dict(data)
    out_data["bboxes_px_per_polyline"] = (
        bboxes  # list of [xmin,ymin,xmax,ymax] (or empty list)
    )
    out_data["polylines_px_resampled"] = resampled
    out_data["n_points_per_line"] = n_points

    out_path = out_dir / path.name

    # # separators to avoid trailing spaces
    # out_path.write_text(json.dumps(out_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)


def process_labels_for_dir(labels_dir: Path, out_dir: Path, n_points):
    json_files = sorted(labels_dir.glob("pos_*.json"))
    for p in json_files:
        process_label_file(p, out_dir, n_points)
    print(f"Done. Wrote {len(json_files)} files to: {out_dir.resolve()}")


dir_name = "test_256"
main_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{dir_name}")
labels_dir = main_dir / "labels"
out_dir = main_dir / "labels_processed"
out_dir.mkdir(parents=True, exist_ok=True)
n_points = 20

# label_file = Path("/home/fatemeh/Downloads/hedge/results/test_mini/labels/pos_000000.json")
# process_label_file(label_file, out_dir, n_points)

process_labels_for_dir(labels_dir, out_dir, n_points)
