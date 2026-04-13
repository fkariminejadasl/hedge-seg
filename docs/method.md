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

### Backbone

- [DINOv3](https://arxiv.org/pdf/2508.10104)
- [PE (Perception Encoder)](https://arxiv.org/pdf/2504.13181)

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


### Others
- [DiffusionEdge](https://arxiv.org/pdf/2401.02032) Diffusion Probabilistic Model for Crisp Edge Detection. Condition on the image to get image images using diffusion model.

<!-- ================================= -->
## Data Sources
- [Data](https://essd.copernicus.org/articles/17/3641/2025) from AHN4 (GeoTIFF).
- TOPO10NL: Ground truth data for hedge, tree, road, and building. Data is provided by PDOK platform. There is also Germany: ATKIS, Great Britain: Mastermap, Denmark: TOP10DK in Chapter 6 https://kadaster.github.io/imbrt .
- [Beeldmateriaal aerial images](https://www.beeldmateriaal.nl/bekijk-luchtfotos)
- [Map2ImLas](https://doi.org/10.1016/j.ophoto.2025.100112): Large-scale 2D-3D airborne dataset with map-based annotations.

PDOK aerial images
- [download data](https://www.beeldmateriaal.nl/dataroom)
- [viewer](https://app.pdok.nl/viewer)
- QGIS: WMTS https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts
<!-- # 194297  408398 -->


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
        