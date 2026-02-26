# Description

## Data

All LineString, point [2, 184], no empty, no invalid, all simple (no self crossing), closed=ring 318 out of 62,415 items, 2 pts 13,307.
In 256x256 image the max number of polylines are 441 and per polyline we take 20 points. 


## Scripts

`build_training_data.py`: create images and labels (`training_data`), postprocess labels (`label_postprocess`), compute embeddings and Combine embeddings with (post-processed) labels (`embeddings_and_pack`).