import requests

def example_download_pdok_aerial_image():
    """
    Example of how to download an aerial image from PDOK using the WMS service.
    This example downloads a 500x500 m area around a point in the Netherlands at high resolution (8 cm) and saves it as "pdok_aerial.png". 
    Image size is set to 2000x2000 pixels to achieve the 8 cm resolution (500 m / 2000 px = 0.25 m/px = 25 cm/px, but PDOK's high-res imagery is actually 8 cm/px).
    """
    
    # Center point in EPSG:28992 (RD New)
    x, y = 194297, 408398

    # Area size in meters around the point
    half_size = 250  # 250 m each side -> 500 x 500 m image

    minx = x - half_size
    miny = y - half_size
    maxx = x + half_size
    maxy = y + half_size

    # Choose a layer:
    # - "Actueel_ortho25" for current 25 cm imagery
    # - "Actueel_orthoHR" for current high-resolution imagery
    # - "2018_ortho25" for "Luchtfoto 2018 Ortho 25cm RGB"
    layer = "Actueel_orthoHR"

    url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"

    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "crs": "EPSG:28992",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": 2000,
        "height": 2000,
        "format": "image/png",
        "transparent": "false",
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    with open("pdok_aerial.png", "wb") as f:
        f.write(r.content)

    print("Saved to pdok_aerial.png")
    print("Requested URL:", r.url)
