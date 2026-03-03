import json


def get_n_polylines_from_json(json_dir):
    n_lines_dic = dict()
    for json_path in json_dir.glob("*.json"):
        with json_path.open("r") as f:
            data = json.load(f)
        n_lines = data.get("n_lines")  # None if missing
        n_lines_dic[json_path.stem] = n_lines
    max_lines = max(n_lines_dic.values())
    max_filename = [n for n, v in n_lines_dic.items() if v == max_lines]
    print(f"Max n_lines: {max_lines} in file(s): {max_filename}")
    return n_lines_dic


"""
from pathlib import Path
folder = "test_mini3"  # "test_mini3" #"test_256" #"test_dataset"
json_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{folder}/labels")
n_lines_dic = get_n_polylines_from_json(json_dir)
print("Done")
"""
