import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize


def segmentation_to_edges(seg):
    # Compute horizontal and vertical differences
    edge_h = np.zeros_like(seg, dtype=bool)
    edge_v = np.zeros_like(seg, dtype=bool)

    edge_h[:, 1:] = seg[:, 1:] != seg[:, :-1]
    edge_v[1:, :] = seg[1:, :] != seg[:-1]

    edge = edge_h | edge_v
    return edge.astype(np.uint8)


def segmentation_to_edges_per_object(seg, background_label=0):
    """
    seg: 2D array with integer labels
    background_label: label to ignore when generating per object edges
    returns: dict {label_value: edge_mask}
    """
    labels = np.unique(seg)
    labels = labels[labels != background_label]

    edges_per_obj = {}

    for lbl in labels:
        # mask for this object
        obj_mask = (seg == lbl).astype(np.uint8)
        # edges only for this object
        edges_per_obj[lbl] = segmentation_to_edges(obj_mask)

    return edges_per_obj


def segmentation_to_edges_stack(seg, background_label=0):
    labels = np.unique(seg)
    labels = labels[labels != background_label]

    edge_stack = np.zeros((len(labels),) + seg.shape, dtype=np.uint8)

    for i, lbl in enumerate(labels):
        obj_mask = (seg == lbl).astype(np.uint8)
        edge_stack[i] = segmentation_to_edges(obj_mask)

    return labels, edge_stack


# Create a blank segmentation map
seg = np.zeros((200, 200), dtype=np.uint8)

# Add some shapes as segmentation classes
cv2.rectangle(seg, (20, 20), (80, 180), 1, -1)  # Class 1
cv2.circle(seg, (140, 60), 40, 2, -1)  # Class 2
cv2.circle(seg, (140, 140), 30, 3, -1)  # Class 3

# Show the segmentation map
plt.figure()
plt.imshow(seg, cmap="tab20")
plt.title("Synthetic Segmentation Mask")
plt.axis("off")
plt.show(block=False)
# Convert segmentation to edges
edges = segmentation_to_edges(seg)
seg_edges = segmentation_to_edges_per_object(seg)
labels, edge_stack = segmentation_to_edges_stack(seg)
# Show the edges
plt.figure()
plt.imshow(seg_edges[1], cmap="gray")
plt.show(block=False)


gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 190, 200)

# Find connected components
num_labels, labels = cv2.connectedComponents(edges)
plt.figure()
plt.imshow(labels == 1, cmap="gray")
plt.show(block=False)
# Create masks for each object
objects = []
for i in range(1, num_labels):  # skip background (label 0)
    mask = np.uint8(labels == i) * 255
    objects.append(mask)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [cnt], -1, 255, 1)

# Skeletonize edges
thin_edge = skeletonize(edges > 0)  # binary
dilated_edge = binary_dilation(edges, iterations=1)

"""
UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif

conda install -c conda-forge gdal

gdalinfo "/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif"

# Convert float32 -> UInt16 with automatic scaling based on min/max in the data
gdal_translate -ot UInt16 -scale "/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif" "/home/fatemeh/Downloads/knepp_95_height_u16.tif"

# Or with explicit scaling (example: expected range 0..50 meters -> full 16-bit)
gdal_translate -ot UInt16 -scale 0 50 0 65535 "/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif" "/home/fatemeh/knepp_95_height_u16.tif"
"""
