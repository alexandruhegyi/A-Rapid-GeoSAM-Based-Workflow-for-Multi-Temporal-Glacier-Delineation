from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import rasterio
from rasterio.transform import xy as pix2xy
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
import geopandas as gpd
from shapely.geometry import box, Point


# =========================
# CONFIG
# =========================
ROOT = Path.home() / "munte/GEOSAM_Svalbard"
INPUTS_DIR = ROOT / "inputs"
PROMPTS_DIR = ROOT / "prompts"

CONNECTIVITY = 2          # 1=4-neighbor, 2=8-neighbor
MIN_AREA_M2 = 5_000       # skip tiny blobs

POS_POINTS_PER_BLOB = 3
NEG_POINTS_PER_BLOB = 2

# Negative "ring" around each blob, in meters (EPSG:25833)
NEG_BUFFER_M = 60.0       # how far outside to look
NEG_RING_WIDTH_M = 80.0   # thickness of the ring

BOX_PADDING_M = 100.0

RANDOM_SEED = 42


# =========================
# Helpers
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_mask(mask_path: Path):
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
        mask = arr.astype(np.uint8) > 0
        resx = abs(transform.a)
        resy = abs(transform.e)
        px_area = resx * resy
        return mask, transform, crs, (resx, resy), px_area


def make_box_with_padding(bounds, pad_m: float):
    minx, miny, maxx, maxy = bounds
    return box(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def sample_points_from_pixels(pixel_coords: np.ndarray, n: int, rng: random.Random):
    """pixel_coords is Nx2 array of (row, col)."""
    if pixel_coords.shape[0] == 0:
        return []
    if pixel_coords.shape[0] <= n:
        idx = list(range(pixel_coords.shape[0]))
        rng.shuffle(idx)
        idx = idx[:n]
    else:
        idx = rng.sample(range(pixel_coords.shape[0]), n)
    return pixel_coords[idx]


def process_year(year_dir: Path, rng: random.Random):
    year = year_dir.name
    mask_path = year_dir / "ICE_MASK.tif"
    if not mask_path.exists():
        print(f"[WARN] {year}: missing ICE_MASK.tif")
        return

    mask, transform, crs, (resx, resy), px_area = read_mask(mask_path)

    lbl = label(mask, connectivity=CONNECTIVITY)
    props = regionprops(lbl)

    boxes_rows = []
    points_rows = []

    # Convert meters to approximate pixels (use mean resolution)
    mean_res = float((resx + resy) / 2.0)
    buf_px = max(1, int(round(NEG_BUFFER_M / mean_res)))
    ring_px = max(1, int(round(NEG_RING_WIDTH_M / mean_res)))

    blob_id = 0
    for prop in props:
        area_m2 = prop.area * px_area
        if area_m2 < MIN_AREA_M2:
            continue

        blob_id += 1
        minr, minc, maxr, maxc = prop.bbox

        # bbox to map coords
        x1, y1 = pix2xy(transform, minr, minc, offset="ul")
        x2, y2 = pix2xy(transform, maxr, maxc, offset="lr")
        minx, maxx = (min(x1, x2), max(x1, x2))
        miny, maxy = (min(y1, y2), max(y1, y2))

        bbox_geom = make_box_with_padding((minx, miny, maxx, maxy), BOX_PADDING_M)
        boxes_rows.append({
            "year": int(year),
            "blob_id": blob_id,
            "area_m2": float(area_m2),
            "pad_m": float(BOX_PADDING_M),
            "geometry": bbox_geom
        })

        # Component mask window
        comp = (lbl[minr:maxr, minc:maxc] == prop.label)

        # Positive points: choose pixels inside the component
        pos_pixels = np.argwhere(comp)  # rows, cols in window coords
        pos_samples = sample_points_from_pixels(pos_pixels, POS_POINTS_PER_BLOB, rng)

        for i, (rr, cc) in enumerate(pos_samples, start=1):
            # convert window pixel -> full raster pixel
            r_full = minr + int(rr)
            c_full = minc + int(cc)
            x, y = pix2xy(transform, r_full, c_full, offset="center")
            points_rows.append({
                "year": int(year),
                "blob_id": blob_id,
                "pt_id": i,
                "label": 1,
                "kind": "pos",
                "geometry": Point(float(x), float(y))
            })

        # Negative points: create a ring around the component in pixel space
        # ring = dilate(comp, buf_px + ring_px) - dilate(comp, buf_px)
        outer = binary_dilation(comp, footprint=disk(buf_px + ring_px))
        inner = binary_dilation(comp, footprint=disk(buf_px))
        ring = outer & (~inner)

        neg_pixels = np.argwhere(ring)
        neg_samples = sample_points_from_pixels(neg_pixels, NEG_POINTS_PER_BLOB, rng)

        # Fallback if ring is empty (small blobs): sample from padded bbox corners
        if len(neg_samples) == 0:
            minx2, miny2, maxx2, maxy2 = bbox_geom.bounds
            fallback = [
                Point(minx2 + 5, miny2 + 5),
                Point(minx2 + 5, maxy2 - 5),
                Point(maxx2 - 5, miny2 + 5),
                Point(maxx2 - 5, maxy2 - 5),
            ][:NEG_POINTS_PER_BLOB]
            for j, p in enumerate(fallback, start=1):
                points_rows.append({
                    "year": int(year),
                    "blob_id": blob_id,
                    "pt_id": j,
                    "label": 0,
                    "kind": "neg",
                    "geometry": p
                })
        else:
            for j, (rr, cc) in enumerate(neg_samples, start=1):
                r_full = minr + int(rr)
                c_full = minc + int(cc)
                x, y = pix2xy(transform, r_full, c_full, offset="center")
                points_rows.append({
                    "year": int(year),
                    "blob_id": blob_id,
                    "pt_id": j,
                    "label": 0,
                    "kind": "neg",
                    "geometry": Point(float(x), float(y))
                })

    # Write outputs
    if boxes_rows:
        gdf_boxes = gpd.GeoDataFrame(boxes_rows, geometry="geometry", crs=crs)
        out_boxes = PROMPTS_DIR / f"{year}_boxes.geojson"
        gdf_boxes.to_file(out_boxes, driver="GeoJSON")
        print(f"[OK] {year}: wrote {len(gdf_boxes)} boxes -> {out_boxes.name}")
    else:
        print(f"[WARN] {year}: no boxes written")

    if points_rows:
        gdf_points = gpd.GeoDataFrame(points_rows, geometry="geometry", crs=crs)
        out_points = PROMPTS_DIR / f"{year}_points.geojson"
        gdf_points.to_file(out_points, driver="GeoJSON")
        print(f"[OK] {year}: wrote {len(gdf_points)} points -> {out_points.name}")
    else:
        print(f"[WARN] {year}: no points written")


def main():
    ensure_dir(PROMPTS_DIR)
    rng = random.Random(RANDOM_SEED)

    year_dirs = sorted([p for p in INPUTS_DIR.iterdir() if p.is_dir() and p.name.isdigit()])
    if not year_dirs:
        raise RuntimeError(f"No year folders found in {INPUTS_DIR}")

    print("Found years:", ", ".join([p.name for p in year_dirs]))

    for yd in year_dirs:
        process_year(yd, rng)

    print("\nDone. Prompts written to:", PROMPTS_DIR)


if __name__ == "__main__":
    main()

