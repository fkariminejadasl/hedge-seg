# QGIS

## Install QGIS

Follow the official installation in [QGIS](https://qgis.org/resources/installation-guide/#debian--ubuntu).

## QGIS Setup

- Shape file: just drage and drop
- GeoTIFF file: just drage and drop
- PDOK aerial images: WMTS https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts
- PDOK AHN: WMTS https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/wms/v1_0
- PDOK TOP10NL: WMTS https://service.pdok.nl/brt/top10nl/wmts/v1_0?request=GetCapabilities&service=wmts
- Satellite NL: WMTS https://wmts.satellietdataportaal.nl/wmts/SuperView-NEO-2026-4-RGB/service?SERVICE=WMTS&REQUEST=GetCapabilities . Get URL from `exps/download_dutch_satellite_data.py` and then add `?SERVICE=WMTS&REQUEST=GetCapabilities`.
- OpenStreetMap: XYZ Tiles https://tile.openstreetmap.org/{z}/{x}/{y}.png
- ESRI World Imagery (Satellite): XYZ Tiles https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
- ESRI World Topo: XYZ Tiles https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}

#### Handy URL

- TOP10NL Documentation: https://kadaster.github.io/imbrt
- TOP10NL Service root: https://api.pdok.nl/brt/top10nl/ogc/v1
- Example TOP10NL: https://api.pdok.nl/brt/top10nl/ogc/v1/collections/inrichtingselement_lijn
- TOP10NL Dataset: https://www.pdok.nl/atom-downloadservices/-/article/basisregistratie-topografie-brt-topnl
- QGIS documentation: https://docs.qgis.org/latest/en/docs/user_manual/working_with_ogc/ogc_client_support.html

---
## Data sources

#### Satellite / Aerial
- Pleiades (paid and free): best resolution 30cm
- NAIP (National Agriculture Imagery Program): aerial 30cm, frequency every 2–3 years
- Maxar (paid): 30 cm
- Planet (paid): 3m, 
- Sentinel: best resolution 10m
- Landsat: best resolution 15m, frequency every 8–16 days
- MODIS which operates on NASA's Terra and Aqua satellites: low res but covers the entire planet every 1 to 2 days. 250m - 1km, daily frequency

[MMEarth-Bench](https://arxiv.org/html/2602.06285) a collection of five new multimodal environmental tasks with 12 modalities, globally distributed data, and both in- and out-of-distribution test splits.

---
## PDOK TOP10NL Specific Layer

In TOP10NL, **“heg, haag”** is an object type inside the **objectklasse `inrichtingselement`**, and it is stored as a **line geometry**. PDOK offers TOP10NL both as an **OGC API Features** service and as downloadable **GeoPackage/GML** files. QGIS supports **WFS / OGC API Features** connections, so you can use either route.

The easiest workflow is this:

#### Option 1: download TOP10NL and export only hedges

1. Open the PDOK TOPNL dataset page and choose the **TOPNL landelijke download (GeoPackage en GML)**. PDOK explicitly states that TOPNL can be downloaded as **GeoPackage and/or GML** via the Atom download service. 

2. Download the **TOP10NL GeoPackage** version.
   GeoPackage is usually easier than GML in QGIS.

3. In QGIS, add the GeoPackage layer that corresponds to **`inrichtingselement_lijn`**.

4. Open the attribute table and filter on the field:

   * `typeinrichtingselement = 'heg, haag'`

   According to the BRT catalog, `heg, haag` is indeed a valid value of `IE_typeInrichtingselement`, and it is line-based. ([Kadaster][1])

5. Right-click the filtered layer → **Export** → **Save Features As...**

6. Set:

   * **Format**: `ESRI Shapefile`
   * **File name**: for example `heg_haag_nl.shp`
   * **CRS**: preferably **EPSG:28992** if you want Dutch RD coordinates

7. Save.

That gives you a shapefile with only the Dutch hedge/haag features.

#### Option 2: direct from PDOK OGC API into QGIS, then export to shapefile

PDOK has a TOP10NL **OGC API Features** endpoint, and the relevant collection is **`inrichtingselement_lijn`**. The collection metadata lists keywords including **heg, haag**, and the collection was last updated on **2026-02-04**. 

Use this service root in QGIS:

```text
https://api.pdok.nl/brt/top10nl/ogc/v1
```

Then in QGIS:

1. **Layer** → **Add Layer** → **Add WFS / OGC API - Features Layer**
2. Create a **New** connection
3. URL:

   ```text
   https://api.pdok.nl/brt/top10nl/ogc/v1
   ```
4. Connect
5. Choose the layer **`inrichtingselement_lijn`**
6. Add it to the map
7. Filter with:

   ```sql
   "typeinrichtingselement" = 'heg, haag'
   ```
8. Right-click layer → **Export** → **Save Features As...** → **ESRI Shapefile**

QGIS supports **OGC API - Features** through the same client used for WFS.

---
## Create a New Field in Attribute Table

For a shape file, the new field can be added to the attribute table. Below is the example:

Do this:

1. Open your layer’s **Attribute Table**.
2. Click **Field Calculator**.
3. Choose one of these:

   * **Create a new field** if you want to store the WKT in the file/table
   * **Create virtual field** if you only want it inside the QGIS project and not written back to the datasource. QGIS expressions explicitly support creating virtual fields.
4. Set:

   * field name: `wkt`
   * field type: **Text / String**
5. In the expression box, enter:

   ```qgis
   geom_to_wkt($geometry)
   ```

   That function converts the feature geometry to WKT text.
6. Click **OK** and save edits if it is a real field.

A very important detail: the `wkt` field may get **cut off** if the geometry text is long. Shapefile attributes use dBASE, and text fields are limited to **254 characters**. Complex lines and polygons often produce WKT strings much longer than that.

---
## Create a bounding box in QGIS

Create a perfect rectangle in QGIS, save it as a CSV, and later load that CSV back into QGIS as a polygon.

#### Step 1: Draw the rectangle in QGIS

1. Create a polygon layer:

   * **Layer → Create Layer → New Temporary Scratch Layer**
   * Geometry type: **Polygon**
   * CRS: choose the CRS you want to work in
   * Click **OK**

2. Select the new layer in the Layers panel.

3. Turn on editing:

   * Right-click the layer → **Toggle Editing**

4. Turn on the rectangle tools:

   * **View → Toolbars → Shape Digitizing Toolbar**

5. Draw a rectangle:

   * Choose **Add Rectangle from 2 Points**
   * Click one corner of the rectangle
   * Click the opposite corner

6. Save edits:

   * Right-click layer → **Toggle Editing**
   * When asked, click **Save**


#### Step 2: Export the rectangle directly as WKT CSV

Right-click the rectangle layer and choose:

```text
Export → Save Features As...
```

Use these settings:

```text
Format: Comma Separated Value [CSV]
File name: bbox_wkt.csv
CRS: same CRS as your rectangle layer
Geometry: AS_WKT
```

Then click **OK**.

This should directly create a CSV like:

```csv
WKT
"POLYGON ((158876.70955029 458870.742176458, 159894.594308784 458870.742176458, 159894.594308784 460081.988555741, 158876.70955029 460081.988555741, 158876.70955029 458870.742176458))"
```

That is the important part: **export with Geometry = AS_WKT**.


#### Step 3: Load the CSV back into QGIS

Drag and drop directly works. For manual upload:

```text
Layer → Add Layer → Add Delimited Text Layer
```

Then set:

```text
File: bbox_wkt.csv
Geometry definition: Well known text (WKT)
Geometry field: WKT
Geometry CRS: same CRS as the original rectangle layer
```

Click **Add**.

The rectangle should appear.

#### Summary

The direct workflow is:

```text
Draw rectangle polygon → Export layer as CSV → Geometry = AS_WKT → Load CSV as WKT
```

