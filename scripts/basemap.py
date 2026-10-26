"""Dark basemap tiles for the maps (Esri « World Dark Gray Base » canvas, the basemap of the 2024 figures), fetched on demand
into a cache outside the repository and composed into one image in Web Mercator. Tiles are credited on the map and
never stored in the repository."""
from __future__ import annotations
import math, io, urllib.request
from pathlib import Path
import numpy as np

CACHE = Path.home() / '.cache' / 'montreal-collisions' / 'tiles'
URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'
ATTRIBUTION = 'Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community'
R = 6378137.0


def merc(lon, lat):
    """Longitude/latitude (degrees) to Web Mercator metres (EPSG:3857)."""
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def _tile_bounds_merc(x, y, z):
    n = 2 ** z
    lon0 = x / n * 360 - 180; lon1 = (x + 1) / n * 360 - 180
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    x0, y0 = merc(lon0, lat0); x1, y1 = merc(lon1, lat1)
    return float(x0), float(x1), float(y0), float(y1)


def _fetch(z, x, y):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'esri_dark_{z}_{x}_{y}.png'
    if not f.exists():
        req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={'User-Agent': 'Montreal-collisions static maps (matplotlib)'})
        with urllib.request.urlopen(req, timeout=60) as r:
            f.write_bytes(r.read())
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(f.read_bytes())).convert('RGB')).astype(float) / 255.0


def basemap(lon0, lon1, lat0, lat1, zoom):
    """Return (image array, extent in Web Mercator metres [x0, x1, y0, y1]) covering the box."""
    tx0, ty1 = _tile_xy(lon0, lat0, zoom); tx1, ty0 = _tile_xy(lon1, lat1, zoom)
    xs = range(int(math.floor(tx0)), int(math.floor(tx1)) + 1); ys = range(int(math.floor(ty0)), int(math.floor(ty1)) + 1)
    rows = []
    for y in ys:
        rows.append(np.concatenate([_fetch(zoom, x, y)[:, :, :3] for x in xs], axis=1))
    img = np.concatenate(rows, axis=0)
    X0 = _tile_bounds_merc(xs[0], ys[0], zoom)[0]; X1 = _tile_bounds_merc(xs[-1], ys[-1], zoom)[1]
    Y1 = _tile_bounds_merc(xs[0], ys[0], zoom)[3]; Y0 = _tile_bounds_merc(xs[-1], ys[-1], zoom)[2]
    return img, [X0, X1, Y0, Y1]
