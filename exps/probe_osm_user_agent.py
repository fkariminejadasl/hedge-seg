"""
Why does the OSM tile server send a "blocked" image instead of a map?

`hedge_seg/training_data.py` draws its OSM chips with contextily, which asks
tile.openstreetmap.org for tiles. Each request carries a User-Agent, the name a
program sends to say who it is. contextily sends contextily-<random hex>, which
names no app. This asks for one tile twice, changing only that name.

Result (2026-10-01, contextily 1.7.0):

    user agent                  HTTP   bytes   image
    contextily-<random hex>     200     6,987  says 403 Access blocked
    hedge-seg research probe    200    46,539  the map

So the server blocks the name, not the address. It still answers 200, so
contextily raises nothing and saves the notice as if it were a map. That is why
`tests/test_training_data.py` has OSM off. To use OSM again, pass
headers={"user-agent": ...} to ctx.bounds2img. No account is needed. Policy:
https://operations.osmfoundation.org/policies/tiles/

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_osm_user_agent.py
"""

from pathlib import Path

import cv2
import numpy as np
import requests
from contextily.tile import USER_AGENT


def main(cfg):
    tiles = []
    for ua in (USER_AGENT, cfg["own_user_agent"]):
        r = requests.get(cfg["tile_url"], headers={"user-agent": ua}, timeout=20)
        print(f"{ua:45s} HTTP {r.status_code}  {len(r.content):,} bytes")
        buf = np.frombuffer(r.content, np.uint8)
        tiles.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    gap = np.full((tiles[0].shape[0], 8, 3), 255, np.uint8)
    cv2.imwrite(str(cfg["out_png"]), np.hstack([tiles[0], gap, tiles[1]]))
    print(f"left contextily, right own name: {cfg['out_png']}")


if __name__ == "__main__":
    cfg = dict(
        # a zoom 10 tile over Alphen aan den Rijn
        tile_url="https://tile.openstreetmap.org/10/525/337.png",
        own_user_agent="hedge-seg research probe",
        out_png=Path(
            "/home/fatemeh/Downloads/hedge/screenshots/osm_user_agent_blocked_vs_map.png"
        ),
    )
    main(cfg)
