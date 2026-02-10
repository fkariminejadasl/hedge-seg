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
    # out_data["polylines_px_resampled"] = resampled
    out_data["n_points_per_line"] = n_points

    out_path = out_dir / path.name

    # # separators to avoid trailing spaces
    # out_path.write_text(json.dumps(out_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)


labels_dir = Path("/home/fatemeh/Downloads/hedg/results/test_dataset/labels")
out_dir = Path("/home/fatemeh/Downloads/hedg/results/test_dataset/labels_processed")
out_dir.mkdir(parents=True, exist_ok=True)
np_points = 20

# label_file = Path(
#     "/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/labels/pos_000000.json"
# )
# process_label_file(label_file, out_dir, np_points)


json_files = sorted(labels_dir.glob("pos_*.json"))
for p in json_files:
    process_label_file(p, out_dir, np_points)

print(f"Done. Wrote {len(json_files)} files to: {out_dir.resolve()}")
