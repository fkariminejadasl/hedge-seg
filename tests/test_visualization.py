import numpy as np

from hedge_seg.visualization import _read_polylines


def _write_npz(path, labels=None, scores=None):
    # Three lines, each a constant x so a line can be told apart after filtering.
    polylines = np.stack([np.full((20, 2), x, dtype=float) for x in (10, 20, 30)])
    arrays = {"polylines": polylines}
    if labels is not None:
        arrays["labels"] = np.asarray(labels)
    if scores is not None:
        arrays["scores"] = np.asarray(scores)
    np.savez(path, **arrays)
    return path


def test_labels_stay_with_their_lines_after_score_filter(tmp_path):
    path = _write_npz(tmp_path / "p.npz", labels=[0, 1, 1], scores=[0.99, 0.5, 0.95])
    polylines, labels = _read_polylines(path, score_thresh=0.9)
    assert polylines[:, 0, 0].tolist() == [10, 30]
    assert labels.tolist() == [0, 1]


def test_class_id_keeps_one_class(tmp_path):
    path = _write_npz(tmp_path / "p.npz", labels=[0, 1, 1])
    polylines, labels = _read_polylines(path, class_id=1)
    assert polylines[:, 0, 0].tolist() == [20, 30]
    assert labels.tolist() == [1, 1]


def test_file_without_labels_is_all_hedge(tmp_path):
    path = _write_npz(tmp_path / "p.npz")
    polylines, labels = _read_polylines(path)
    assert len(polylines) == 3
    assert labels.tolist() == [0, 0, 0]
