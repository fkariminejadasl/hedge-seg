# hedge-seg

Detect and extract hedges in vegetation height models derived from **aerial point cloud data**.

---

## 🌿 Overview

This project focuses on extracting **hedges** as curvilinear features from spatially aggregated LiDAR or other point cloud data.  
Each input raster represents a **Vegetation Height Model (VHM)** where each pixel stores the **95th percentile height** of points within a 10 × 10 meter grid cell. In these height surfaces, hedges appear as narrow elongated bands of higher vegetation.

<p align="center">
  <img src="docs/knepp_95_height_u16.jpg" width="400">
</p>

---

## 🎯 Objectives

- Detect hedges in vegetation height surfaces

---

## 🧪 Example Workflow

```python
from hedge_seg import data as hd

# Load vegetation height model (VHM)
image = hd.load_geotiff("UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif")
```

---

## 📏 Terminology

- **Spatial resolution**: size of one pixel on the ground, for example 10 × 10 meters.  
- **Spatial extent**: total area covered by the raster, corresponding to image resolution (height and width in pixels) in deep learning and computer vision.
