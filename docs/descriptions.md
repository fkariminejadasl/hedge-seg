# Description

## Data

All LineString, point [2, 184], no empty, no invalid, all simple (no self crossing), closed=ring 318 out of 62,415 items, 2 pts 13,307

## Scripts


- `generate_training_data.py`: image, map, labels, embeddings.
- `process_labels.py`: to make resample equidistant points on each polyline.
- `format_utils.py`: save embeddings and polylines
- `train_neg_pos_classifier.py`: classifier to identify negative from positive images