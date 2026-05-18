"""
pip install requests shapely geopandas
export SDP_USERNAME="your_email@example.com"
export SDP_PASSWORD="your_password"
"""

import os

import requests
from requests.auth import HTTPBasicAuth

USERNAME = os.environ["SDP_USERNAME"]
PASSWORD = os.environ["SDP_PASSWORD"]

STAC_BASE = "https://api.satellietdataportaal.nl/v2/stac"


auth = HTTPBasicAuth(USERNAME, PASSWORD)

# Wageningen bbox, lon/lat EPSG:4326
BBOX = [5.64, 51.95, 5.72, 52.01]

COLLECTIONS_TO_TRY = [
    "Pleiades-NEO_Nederland",
    "SuperView-NEO_Nederland",
    "SuperView-1_Nederland",
    "PlanetScope_Nederland",
    "RapidEye_Nederland",
]


def search_collection(collection_id, datetime_range):
    payload = {
        "collections": [collection_id],
        "bbox": BBOX,
        "datetime": datetime_range,
        "limit": 10,
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

    # IMPORTANT: this API may return "features": null instead of []
    features = data.get("features") or []

    print(f"Found {len(features)} items")
    return features


def main():
    searches = [
        # Current/recent 30 cm datasets
        ("Pleiades-NEO_Nederland", "2023-02-01T00:00:00Z/2026-05-18T23:59:59Z"),
        ("SuperView-NEO_Nederland", "2023-02-01T00:00:00Z/2026-05-18T23:59:59Z"),
        # 2019 to 2022
        ("SuperView-1_Nederland", "2019-01-01T00:00:00Z/2022-12-31T23:59:59Z"),
        # 2018 only, roughly spring to summer
        ("PlanetScope_Nederland", "2018-04-01T00:00:00Z/2018-09-30T23:59:59Z"),
        # Older RapidEye/Formosat/Spot collections may need older date windows
        ("RapidEye_Nederland", "2017-01-01T00:00:00Z/2018-12-31T23:59:59Z"),
    ]

    all_features = []

    for collection_id, datetime_range in searches:
        features = search_collection(collection_id, datetime_range)
        all_features.extend(features)

    print("\n" + "=" * 80)
    print(f"Total items found: {len(all_features)}")

    for item in all_features:
        print("\nID:", item.get("id"))
        print("Collection:", item.get("collection"))
        print("Datetime:", item.get("properties", {}).get("datetime"))

        assets = item.get("assets") or {}
        for asset_name, asset in assets.items():
            print("  Asset:", asset_name)
            print("    type:", asset.get("type"))
            print("    href:", asset.get("href"))


if __name__ == "__main__":
    main()
