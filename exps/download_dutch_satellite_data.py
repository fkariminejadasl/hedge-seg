"""
Download Satellietdataportaal STAC assets for a selected item.

Install:
    pip install requests

Set credentials:
    export SDP_USERNAME="your_email@example.com"
    export SDP_PASSWORD="your_password"

Run:
    python download_sdp_asset.py
"""

import os
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

USERNAME = os.environ["SDP_USERNAME"]
PASSWORD = os.environ["SDP_PASSWORD"]

STAC_BASE = "https://api.satellietdataportaal.nl/v2/stac"

auth = HTTPBasicAuth(USERNAME, PASSWORD)

# BBOX format: [min_lon, min_lat, max_lon, max_lat]
# Tiny bbox around the point from your viewer URL:
# https://viewer.satellietdataportaal.nl/@52.00401,5.378838,12/
BBOX = [5.3783, 52.0035, 5.3794, 52.0045]

OUT_DIR = Path("/home/fatemeh/Downloads/hedge/results/satellite_downloads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Product you saw in the viewer URL:
TARGET_COLLECTION = "SuperView-NEO_Nederland"
TARGET_DATE_RANGE = "2026-04-29T00:00:00Z/2026-04-29T23:59:59Z"
TARGET_ITEM_ID_PARTS = ["20260429", "SVNEO-04"]

# Other examples:
# # Current/recent 30 cm datasets
# ("SuperView-NEO_Nederland", "2026-04-29T00:00:00Z/2026-04-29T23:59:59Z")
# ("Pleiades-NEO_Nederland", "2025-02-01T00:00:00Z/2026-05-18T23:59:59Z"),
# ("Pleiades-NEO_Nederland", "2023-02-01T00:00:00Z/2026-05-18T23:59:59Z"),
# # 2019 to 2022
# ("SuperView-1_Nederland", "2019-01-01T00:00:00Z/2022-12-31T23:59:59Z"),
# # 2018 only, roughly spring to summer
# ("PlanetScope_Nederland", "2018-04-01T00:00:00Z/2018-09-30T23:59:59Z"),
# # Older RapidEye/Formosat/Spot collections may need older date windows
# ("RapidEye_Nederland", "2017-01-01T00:00:00Z/2018-12-31T23:59:59Z"),

# Choose one:
# "30cm_RGB_8bit_SVNEO"          = visual RGB ZIP
# "30cm_BGRN_11bit_SVNEO"        = multispectral B/G/R/NIR ZIP
# "30cm_BGRN_BOA_32bit_SVNEO"    = BOA corrected multispectral ZIP
# "SVNEO_ruw"                    = raw product ZIP
ASSET_NAME = "30cm_RGB_8bit_SVNEO"


def search_collection(collection_id, datetime_range, bbox):
    payload = {
        "collections": [collection_id],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": 20,
    }

    r = requests.post(
        f"{STAC_BASE}/search",
        auth=auth,
        headers={
            "Accept": "application/geo+json, application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    print("\n" + "=" * 80)
    print("Collection:", collection_id)
    print("Datetime:", datetime_range)
    print("Status:", r.status_code)
    print("Response preview:", r.text[:1000])

    r.raise_for_status()

    data = r.json()
    features = data.get("features") or []

    print(f"Found {len(features)} item(s)")
    return features


def print_item_summary(item):
    print("\n" + "-" * 80)
    print("ID:", item.get("id"))
    print("Collection:", item.get("collection"))
    print("Datetime:", item.get("properties", {}).get("datetime"))
    print("BBOX:", item.get("bbox"))

    print("\nAssets:")
    assets = item.get("assets") or {}
    for name, asset in assets.items():
        print(f"  {name}")
        print(f"    type: {asset.get('type')}")
        print(f"    href: {asset.get('href')}")

    print("\nMap links:")
    for link in item.get("links", []) or []:
        rel = link.get("rel")
        href = link.get("href")
        title = link.get("title")
        link_type = link.get("type")

        if rel in {"xyz", "wmts", "wms", "tilejson"} or "wmts" in (href or "").lower():
            print()
            print("rel  :", rel)
            print("type :", link_type)
            print("title:", title)
            print("href :", href)


def choose_target_item(items):
    matching = []

    for item in items:
        item_id = item.get("id", "")
        if all(part in item_id for part in TARGET_ITEM_ID_PARTS):
            matching.append(item)

    if not matching:
        print("\nNo item matched TARGET_ITEM_ID_PARTS:", TARGET_ITEM_ID_PARTS)
        print("Available item IDs:")
        for item in items:
            print(" ", item.get("id"))
        return None

    if len(matching) > 1:
        print("\nMultiple matching items found. Using the first one:")
        for item in matching:
            print(" ", item.get("id"))

    return matching[0]


def filename_from_response(response, fallback_name):
    content_disposition = response.headers.get("Content-Disposition", "")

    # Simple handling for headers like:
    # Content-Disposition: attachment; filename="something.zip"
    lower = content_disposition.lower()
    if "filename=" in lower:
        filename = content_disposition.split("filename=", 1)[1].strip()
        filename = filename.strip('"').strip("'")
        if filename:
            return filename

    return fallback_name


def download_asset(item, asset_name):
    assets = item.get("assets") or {}

    if asset_name not in assets:
        print(f"\nAsset not found: {asset_name}")
        print("Available assets:")
        for name in assets:
            print(" ", name)
        return None

    url = assets[asset_name].get("href")
    if not url:
        raise ValueError(f"Asset {asset_name} has no href")

    # The STAC response may show http links. Try https first.
    if url.startswith("http://api.satellietdataportaal.nl"):
        url = url.replace("http://", "https://", 1)

    item_id = item.get("id", "sdp_item")
    fallback_name = f"{item_id}_{asset_name}.zip"

    print("\n" + "=" * 80)
    print("Downloading asset")
    print("Item :", item_id)
    print("Asset:", asset_name)
    print("URL  :", url)

    with requests.get(
        url,
        auth=auth,
        stream=True,
        timeout=300,
        allow_redirects=True,
        headers={"Accept": "application/zip,*/*"},
    ) as r:
        print("Status:", r.status_code)
        print("Final URL:", r.url)
        print("Content-Type:", r.headers.get("Content-Type"))
        print("Content-Length:", r.headers.get("Content-Length"))
        print("Content-Disposition:", r.headers.get("Content-Disposition"))

        # Show server message before raising, useful for login/permission errors.
        if r.status_code >= 400:
            preview = r.text[:1000] if r.text else ""
            print("Error response preview:")
            print(preview)

        r.raise_for_status()

        filename = filename_from_response(r, fallback_name)
        out_path = OUT_DIR / filename

        # Avoid overwriting an existing file accidentally.
        if out_path.exists():
            stem = out_path.stem
            suffix = out_path.suffix
            counter = 1
            while True:
                candidate = OUT_DIR / f"{stem}_{counter}{suffix}"
                if not candidate.exists():
                    out_path = candidate
                    break
                counter += 1

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

        bytes_written = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        tmp_path.rename(out_path)

    print("\nSaved:")
    print(out_path)
    print(f"Size: {bytes_written / 1024 / 1024:.2f} MB")

    return out_path


def main():
    items = search_collection(
        collection_id=TARGET_COLLECTION,
        datetime_range=TARGET_DATE_RANGE,
        bbox=BBOX,
    )

    if not items:
        print("\nNo STAC items found for this bbox/date/collection.")
        return

    print("\nItems returned by search:")
    for item in items:
        print(" ", item.get("id"))

    target_item = choose_target_item(items)

    if target_item is None:
        return

    print_item_summary(target_item)

    # Very slow. It took 15 minutes for a 30cm_RGB_8bit_SVNEO.
    downloaded_path = download_asset(target_item, ASSET_NAME)

    if downloaded_path:
        print("\nDone.")
    else:
        print("\nNothing downloaded.")


if __name__ == "__main__":
    main()
