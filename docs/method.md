# Hedge Segmentation

## Method

Preprocessing: image super resolution

Option1: DETR like curvlinear objects on DINO embeddings

Option2: Semantic segementation (SS) or instant segmentation (IS). Segmentation model such as [SAM 3](https://ai.meta.com/blog/segment-anything-model-3) or just DPT head on DINO backbone. Both SS and IS requires postprocessing to get vector data. Instance segmentation postprocessing is simpler than SS.

For better separation of hedge and not confusing between tree, we could treat trees as negatives. We could get the idea from [CountGD++ Zisserman](https://arxiv.org/abs/2512.23351)​, to use it in both data and loss.

Directly tesing it with SAM 3 and PiDiNet (edge detection) didn't result in a satisfactory results.

<!-- ==================================== -->

## Literature

Image to vector models where given image and extract vectors directly. Vector can be polyline (curve), and polygon.

### Backbone / Embeddings

- [DINOv3](https://arxiv.org/pdf/2508.10104) from [collections](https://huggingface.co/collections/facebook/dinov3).
- [PE (Perception Encoder)](https://arxiv.org/pdf/2504.13181)
- Google AlphaEarth / [Satellite Embedding](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL): a precomputed geospatial embedding dataset in which each approximately 10 m × 10 m ground pixel has one 64-dimensional embedding vector for each calendar year. The dataset is available in Google Earth Engine as `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.

### Self Driving

These refernce are the DETR-like model, which from BEV images they get the vector data.

- [MapTRv2](https://arxiv.org/pdf/2308.05736v2), [MapTR](https://openreview.net/pdf/f0aa5f3818d2d071eed47bfd84263b7b217b437a.pdf).
- [FlexMap](https://arxiv.org/pdf/2601.22376)
- [MapQR](https://arxiv.org/pdf/2402.17430)
- [BezierFormer](https://arxiv.org/pdf/2404.16304).
- PolyRoad: Polyline Transformer for Topological Road-Boundary Detection.
- [VectorMapNet]

### Medical Imaging

- DeformCL: Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image. First semantic segmentation, postprocess and get points and then DETR-like architecture for getting curves. Note that features are only points not the embedding of the image as in the DETR memory is the embedding of the whole image up to patch size.

### Computer Graphics

- [NeuralFur](https://arxiv.org/pdf/2601.12481). modeling strand with MLP of root point (MLP(x))

### Remote Sensing

- [Hedgerow mapping with high resolution satellite imagery](https://www.sciencedirect.com/science/article/pii/S0034425725002743)
- [Hedgerow review](https://ieeexplore.ieee.org/document/10731836)

### Others

- [DiffusionEdge](https://arxiv.org/pdf/2401.02032) Diffusion Probabilistic Model for Crisp Edge Detection. Condition on the image to get image images using diffusion model.
- UNIGEOCLIP: Unified Geospatial Contrastive Learning
- MMEarth-Bench: Global Environmental Tasks for Multimodal Geospatial Models
- Entropy-Gradient Grounding: Training-Free Evidence Retrieval in Vision-Language Models. Just an idea for fine-grained information.

<!-- ================================= -->

## Data Sources

#### Netherlands

- [Data](https://essd.copernicus.org/articles/17/3641/2025) from AHN4 (GeoTIFF).
- TOPO10NL: Ground truth data for hedge, tree, road, and building. Data is provided by PDOK platform. There is also Germany: ATKIS, Great Britain: Mastermap, Denmark: TOP10DK in Chapter 6 https://kadaster.github.io/imbrt .
- [Beeldmateriaal aerial images](https://www.beeldmateriaal.nl/bekijk-luchtfotos)
- [Map2ImLas](https://doi.org/10.1016/j.ophoto.2025.100112): Large-scale 2D-3D airborne dataset with map-based annotations.

PDOK aerial images (only Netherlands)

- [download data](https://www.beeldmateriaal.nl/dataroom)
- [viewer](https://app.pdok.nl/viewer)
- QGIS: WMTS https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts

<!-- # 194297  408398 -->

[Satellietdataportaal](https://www.satellietdataportaal.nl), NLSA (only Netherlands):

- PlanetScope (4.8m), RapidEye(6m), Pleiades NEO (30cm, 50cm), SuperView-1(50cm), TripleSat(80cm), Hyperscout-2, Sentinel-2(10m), Spot6-7(1.5m), Formosat(2m), RadarSat-2(20m)

- Copernicus Data Space Ecosystem or ESA Earth Online:
  Sentinel / Copernicus / ESA data

For the Netherlands, two main sources of high-resolution remote sensing data are used. Satellite imagery is obtained from the Satellietdataportaal of the Netherlands Space Agency, which provides SuperView data at approximately 30 cm resolution. Each tile covers about 14 × 14 km and is around 6.5 GB, with new imagery generally available about once per month for the same location, although cloud cover can affect usability. Aerial imagery is available through the PDOK platform and originates from Beeldmateriaal Nederland, a Dutch public-sector initiative implemented by Het Waterschapshuis and Kadaster. These aerial images are higher quality and are available at 8 cm and 25 cm resolution, typically updated annually.

#### Others

- [Mapping and classification of trees outside forests](https://www.sciencedirect.com/science/article/pii/S2666017226001483): 9,000 unique 1024×1024 RGB aerial crops at 20 cm/pixel from 4 regions × 100 tiles = 400 original image tiles. Classes: forest, patch, linear (hedgerow + treelines), tree. Each original tile is 5000×5000 pixels, covering 1 km × 1 km; training tiles are cut into non-overlapping 1024×1024 crops. H/V flips are saved on disk, giving 27,000 training samples from the 9,000 crops. 96M-parameter FT-UNetFormer model. Reference labels were automatically generated from nDSM/NDVI and manually refined.
- [From Pixels to Planning: Earth AI for Nature Restoration](https://research.google/blog/from-pixels-to-planning-earth-ai-for-nature-restoration/) (Google Research, 2026): High-resolution mapping of hedgerows, woodland, and stone walls across England (~130,000 km²). Uses a ViT backbone pretrained on 300M+ satellite images (resolution/size unspecified), Fine-tuned on 942 manually annotated aerial RGB tiles (25 cm/pixel, 2048 × 2048 pixels, 512 × 512 m), split into 742/100/100 train/val/test, using random 512 × 512 pixel crops (128 × 128 m) for fine-tuning. Produces 25 cm resolution raster probability maps and vector polygons with 5 classes: hedgerow, copse, linear woodland, stone wall, and woodland. Available via Google Earth Engine: [Raster dataset](https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_farmscapes_england_v1_0) contains 3 model-predicted probability layers (hedgerow, stone wall, woodland/tree); [Vector dataset](https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_farmscapes_england_v1_0_vectorised) contains post-processed polygons, derived from raster predictions using geometric filtering and shape-based classification (not direct model outputs). Original aerial RGB imagery and manually annotated training data are not publicly provided with the datasets. Aerial imagery could be obtained separately and paired with the predictions, subject to geographic alignment, acquisition dates, and licensing. [Technical paper](https://arxiv.org/html/2506.13993).
- [NEST3D](https://arxiv.org/html/2606.14562v1) from Tuia: 104 nest-bearing trees, 27,945 RGB + 111,780 4-band multispectral images (Green/Red/Red Edge/NIR), reconstructed into ~951.7M 3D points. Point-level 3D labels: grass, tree, nest, manually annotated in CloudCompare; 72/16/16 train/val/test split. Benchmarked PT-v3, RandLA-Net, KPConv; PT-v3 reaches 90.0% mIoU. Open on Hugging Face.
- [GroundSet](https://arxiv.org/abs/2603.14609) (ECCV 2026): 510k high-resolution aerial RGB images (20 cm/pixel, 672 × 672 pixels, ~134 × 134 m), 3.8M annotated objects across 135 classes, grounded in French cadastral vector data (IGN). Supports 7 spatial understanding tasks, including detection, segmentation, captioning, and VQA. Fine-tuned LLaVA outperforms specialized remote sensing and commercial models. Dataset, model, and code available on Hugging Face. [Class taxonomy](https://github.com/rogerferrod/GroundSet/blob/main/src/resources/tree.json) Includes Vegetation, Forest, Woodland, Orchard, and Vineyard, but no explicit hedgerow or tree-row classes. [research.google](https://research.google/pubs/groundset-a-cadastral-grounded-dataset-for-spatial-understanding-with-vector-data)
- [Align and Segment (AnS)](https://arxiv.org/abs/2607.10841) (ECCV 2026): Self-supervised building segmentation from misaligned labels (e.g., OpenStreetMap). Jointly trains a segmentation network and spatial transformer to correct labels via image-level affine transformations, without requiring aligned ground-truth labels. Evaluated on synthetic and real-world datasets (41 cities); outperforms unsupervised baselines. Code, datasets, and pretrained weights available on [GitHub](https://github.com/venkanna37/align-and-segment). [mosquito-risk.github.io](https://mosquito-risk.github.io/align-and-segment)

-----

- [reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis](https://arxiv.org/abs/2407.03653) (2024): Improved version of BigEarthNet containing 549,488 Sentinel-1 SAR + Sentinel-2 multispectral image pairs (10–60 m/pixel, 1.2 × 1.2 km per patch, 120 × 120 pixels at 10 m). Fixes noisy labels, improves atmospheric correction, and introduces geographically separated train/val/test splits. Includes 19 land-cover classes, with image-level labels and pixel-level segmentation reference maps automatically derived from updated CORINE Land Cover 2018, without manually annotating individual images. Limitation: CORINE has a minimum mapping unit of 25 hectares (equivalent in area to 500 × 500 m) and a minimum feature width of 100 m, meaning narrow features like hedgerows are generally not individually labeled. Additionally, Sentinel-2's 10 m resolution is too coarse for precise hedgerow segmentation. Satellite images, annotations, code, and pretrained models are publicly available. [Dataset](https://bigearth.net/). [arXiv](https://arxiv.org/html/2407.03653)
- [VegCAMP: Vegetation Classification and Mapping Program](https://wildlife.ca.gov/Data/VegCAMP) (California Department of Fish and Wildlife): Regional vegetation maps with detailed classes including forest, woodland, shrubland, grassland, riparian vegetation, and wetlands. Uses expert interpretation of aerial imagery and field surveys to produce manually delineated GIS polygons, classified according to NVCS (class count varies by project). Aerial imagery is available separately through [NAIP](https://earthexplorer.usgs.gov/) (RGB + NIR, typically 0.6–1 m/pixel, no fixed image dimensions). Limitation: Minimum mapping unit typically 1–2 acres (~4,047–8,094 m²), with no consistent hedgerow/tree-row class, making annotations unsuitable for precise hedgerow segmentation. [Vegetation GIS data](https://wildlife.ca.gov/Data/VegCAMP/Reports-and-Maps).

---
- Old review paper on GeoFM: [GeoFM: How Will Geo-Foundation Models Reshape Spatial Data Science and GeoAI?](https://doi.org/10.1080/13658816.2025.2543038) (2025): Review paper defining geospatial foundation models and examining existing models, datasets, and benchmarks, including:
  - [SkySense](https://arxiv.org/abs/2312.10115): 2.06B parameters, including 654M (high-resolution RGB encoder, Swin-H), 302M (Sentinel-2 encoder, ViT-L), 302M (Sentinel-1 SAR encoder, ViT-L), 398M (multimodal temporal fusion), 215M (geo-context prototypes), and 189M (other components). Pretrained on high-resolution satellite RGB imagery (0.3 m/pixel, 2048 × 2048 pixels), Sentinel-2 multispectral (10 m/pixel, 10 bands: RGB, Red Edge, NIR, SWIR), and Sentinel-1 SAR (10 m/pixel, VV/VH radar). Uses multimodal spatiotemporal learning. Pretraining required 24,600 A100 GPU-hours (~1,025 GPU-days or 13 days on 80 A100). Pretrained weights available on [GitHub](https://github.com/Jack-bo1220/SkySense).
  - [CROMA](https://arxiv.org/abs/2311.00566): ~194M (Base) / ~670M (Large) parameters. Pretrained on Sentinel-1 SAR (VV/VH radar) + Sentinel-2 multispectral imagery (12 optical bands, native resolutions 10–60 m, resampled to a common grid).
  - [Prithvi-EO](https://arxiv.org/abs/2412.02732): 100M (v1), 300M / 600M (v2) parameters. Pretrained on NASA HLS imagery (30 m/pixel, 6 bands: RGB, NIR, SWIR1, SWIR2).
  - [SatMAE](https://arxiv.org/abs/2207.08051): ~86M (Base) / 303M (Large) parameters. Pretrained on RGB satellite imagery (variable resolution) and Sentinel-2 multispectral imagery (10–20 m/pixel, 10 bands: RGB, Red Edge, NIR, SWIR; bands resampled to a common grid).
Discusses self-supervised pretraining, multimodal geospatial learning, spatial reasoning, and transfer to downstream applications such as land-cover mapping. No new model or dataset introduced. [SatMAE code](https://github.com/sustainlab-group/SatMAE) | [Prithvi models](https://huggingface.co/ibm-nasa-geospatial) | [CROMA code](https://github.com/antofuller/CROMA).

- [SAGE: A Sampling-Aware Global Evaluation Benchmark for Species Distribution Modeling](https://arxiv.org/html/2609.31082) (Tuia, 2026): Global benchmark for predicting the geographic distribution of 5,771 plant species. Combines 89.8M species occurrence records from GBIF (citizen-science observations) for training with 53,336 expert-curated vegetation plots from sPlotOpen for presence-absence evaluation. Uses 52 environmental predictors derived from raster datasets (climate, topography, and human activity at 1 km; soil at 250 m), converted into tabular features for species distribution modeling. No RGB/multispectral imagery or pixel-level segmentation annotations. Not directly suitable for hedgerow segmentation, as it focuses on species distribution rather than vegetation boundaries. Benchmarks traditional species distribution models (Random Forest, MaxEnt, etc.) against deep learning models (MLP, ResNet, FT-Transformer), all using tabular inputs. Finds that DeepSDMs outperform traditional methods for infrequently recorded species when sampling bias is corrected, but offer no consistent advantage for well-sampled species. Data, code, models, and visualizations available on the [project website](https://earens.github.io/sage/).

<!-- ================================= -->

## Issues and some possible solutions

- Data:
  - Short-length data (minimum 10 pixels per 256x256 image) are removed. -> Done
  - Low-quality data. Maybe use the self-driving data. -> Not Done
  - Closed polylines: There are 691,006 polylines and 4,121 closed shapes. Some are not originally closed, but their start and end points are very close. -> Not Done
  - Data leakage. Use data from different locations for training and validation. -> Not Done
- Model and Loss:
  - Auxiliary loss:
    - Diffusion model on coordinates. Add an extra decoder to inject noise into the coordinates and denoise them with classifier-free guidance.
    - Semantic segmentation decoder as an auxiliary loss to help the model converge better.
    - Auxiliary loss on every decoder layer, similar to DETR.
    - Cardinality (polyline count), length, direction, and curvature loss.
  - Use only the semantic segmentation network to show that the data and DINOv3 are fine. This might not be necessary, since the binary background/foreground classifier already shows good results.
  - Train DINOv3 as well: the transformer decoder head acts like an adapter, so there is most likely no need to train DINOv3.
- Optimization and Hyperparameters
  - Gradient clipping
  - Scheduler
  - Batch size (lower), learning rate (lower), EOS (lower)
