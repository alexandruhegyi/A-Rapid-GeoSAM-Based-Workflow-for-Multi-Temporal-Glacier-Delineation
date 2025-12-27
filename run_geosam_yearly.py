# run_geosam_autogate_yearly.py
# Glacier outlines with SAM + auto-derived physical gates + smoothed vectors
#
# Inputs per year in inputs/<year>/:
#   P25.tif      (6-band: B2,B3,B4,B8,B11,B12)  EPSG:25833
#   NDSI.tif     (float)
#   NDVI.tif     (float)
#   ICE_MASK.tif (0/1)
#
# Prompts:
#   prompts/<year>_boxes.geojson
#   prompts/<year>_points.geojson
#
# Outputs:
#   sam_masks/<year>/glacier_mask_<year>.tif
#   vectors/glaciers_<year>_shp/glaciers_<year>.shp
#   vectors/glaciers_<year>_shp/glaciers_<year>_smoothed.shp

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import shapes
from shapely.geometry import shape
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

try:
    from skimage.morphology import remove_small_objects, remove_small_holes
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False


# =========================
# CONFIG
# =========================
ROOT = Path.home() / "munte/GEOSAM_Svalbard"

INPUTS_DIR  = ROOT / "inputs"
PROMPTS_DIR = ROOT / "prompts"
MODELS_DIR  = ROOT / "models"
MASKS_DIR   = ROOT / "sam_masks"
VECTORS_DIR = ROOT / "vectors"

CHECKPOINT_PATH = MODELS_DIR / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cpu"  # change to "cuda" if torch+cuda is available

# Use ICE_MASK as reference set for auto thresholds and as final restriction
USE_ICE_MASK = True

# Auto-threshold percentiles derived from ICE_MASK pixels
NDSI_BASE_P = 5     # low tail of ice NDSI
B11_BASE_P  = 95    # high tail of ice B11
NDVI_BASE_P = 99    # high tail of ice NDVI

# Margins applied to auto thresholds
NDSI_MARGIN = 0.05      # subtract from NDSI threshold (slightly looser)
B11_MARGIN  = 50.0      # add to B11 max (slightly looser)
NDVI_MARGIN = 0.05      # add to NDVI max (slightly looser)

# Safety clamps
NDSI_MIN = 0.15
NDSI_MAX = 0.80
B11_MIN  = 200.0
B11_MAX  = 2500.0
NDVI_MIN = -0.5
NDVI_MAX = 0.5

# Skip boxes with too little ICE_MASK coverage
MIN_VALID_FRAC_IN_BOX = 0.02

# Raster mask cleanup (optional but recommended)
DO_MORPHOLOGY = True and SKIMAGE_OK
REMOVE_SMALL_OBJECTS_MIN_PIX = 500
REMOVE_SMALL_HOLES_MAX_PIX = 1200

# Polygon filtering (m²) after polygonize and after smoothing
MIN_POLY_AREA_M2 = 50_000

# --- Vector smoothing (meters; EPSG:25833 is meters)
SMOOTH_OUTPUT = True
SMOOTH_BUFFER_M = 15.0   # 10–30m typical at 10m pixels
SIMPLIFY_M = 5.0         # 0 to disable; 3–10m typical

# Test single year (set to None for all years)
ONLY_YEARS: set[str] | None = {"2018", "2020", "2022", "2024"}  # e.g. {"2018"}; set None for all


# =========================
# Helpers
# =========================
def to_u8_reflectance(b: np.ndarray) -> np.ndarray:
    """Fixed stretch for S2 SR reflectance (0..3000) -> uint8 (0..255)."""
    b = np.nan_to_num(b, nan=0.0).astype(np.float32)
    b = np.clip(b, 0, 3000)
    b = (b * (255.0 / 3000.0)).round()
    return b.astype(np.uint8)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def map_bbox_to_pixel_box(transform, bounds, width: int, height: int) -> np.ndarray:
    """Map bbox (minx,miny,maxx,maxy) -> SAM pixel box [x0,y0,x1,y1]."""
    minx, miny, maxx, maxy = bounds
    r0, c0 = rowcol(transform, minx, maxy)
    r1, c1 = rowcol(transform, maxx, miny)

    x0, x1 = sorted([int(c0), int(c1)])
    y0, y1 = sorted([int(r0), int(r1)])

    x0 = int(clamp(x0, 0, width - 1))
    x1 = int(clamp(x1, 0, width - 1))
    y0 = int(clamp(y0, 0, height - 1))
    y1 = int(clamp(y1, 0, height - 1))

    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)

    return np.array([x0, y0, x1, y1], dtype=np.float32)


def map_points_to_pixel(transform, points_gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    labels = []
    for geom, lab in zip(points_gdf.geometry, points_gdf["label"].tolist()):
        x, y = float(geom.x), float(geom.y)
        r, c = rowcol(transform, x, y)
        coords.append([int(c), int(r)])  # (x,y) pixel
        labels.append(int(lab))
    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int64)


def polygonize_mask(mask: np.ndarray, meta: dict, year: str) -> gpd.GeoDataFrame:
    transform = meta["transform"]
    crs = meta["crs"]

    geoms = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if val == 1:
            geoms.append(shape(geom))

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if len(gdf) == 0:
        gdf["year"] = []
        gdf["area_m2"] = []
        return gdf

    gdf["area_m2"] = gdf.area
    if MIN_POLY_AREA_M2 > 0:
        gdf = gdf[gdf["area_m2"] >= MIN_POLY_AREA_M2].copy()

    gdf["year"] = int(year)
    gdf["area_m2"] = gdf.area
    return gdf


def smooth_geometry(g, buffer_m: float, simplify_m: float):
    """
    Smooth by morphological closing: buffer(+d) then buffer(-d),
    then optional simplify.
    """
    if g is None or g.is_empty:
        return g
    try:
        gg = g.buffer(buffer_m).buffer(-buffer_m)
        if simplify_m and simplify_m > 0:
            gg = gg.simplify(simplify_m, preserve_topology=True)
        return gg
    except Exception:
        return g


def smooth_gdf(gdf: gpd.GeoDataFrame, buffer_m: float, simplify_m: float) -> gpd.GeoDataFrame:
    if len(gdf) == 0:
        return gdf
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(lambda geom: smooth_geometry(geom, buffer_m, simplify_m))
    out = out[out.geometry.notnull() & ~out.geometry.is_empty].copy()
    out["area_m2"] = out.area
    if MIN_POLY_AREA_M2 > 0:
        out = out[out["area_m2"] >= MIN_POLY_AREA_M2].copy()
    return out


def find_year_file(year_dir: Path, year: str, kind: str) -> Path:
    """
    Supports clean names (P25.tif, NDSI.tif, NDVI.tif, ICE_MASK.tif)
    and long export names (S2_2018_Aug10_31_...).
    """
    preferred = {
        "P25": ["P25.tif", "p25.tif"],
        "NDSI": ["NDSI.tif", "ndsi.tif"],
        "NDVI": ["NDVI.tif", "ndvi.tif"],
        "ICE_MASK": ["ICE_MASK.tif", "ice_mask.tif"],
    }
    for name in preferred.get(kind, []):
        p = year_dir / name
        if p.exists():
            return p

    pats = {
        "P25": re.compile(rf".*{year}.*p25.*\.tif$", re.IGNORECASE),
        "NDSI": re.compile(rf".*{year}.*ndsi.*\.tif$", re.IGNORECASE),
        "NDVI": re.compile(rf".*{year}.*ndvi.*\.tif$", re.IGNORECASE),
        "ICE_MASK": re.compile(rf".*{year}.*(ice[_-]?mask|mask).*\.tif$", re.IGNORECASE),
    }
    pat = pats[kind]
    cands = [p for p in year_dir.glob("*.tif") if pat.match(p.name)]
    if not cands:
        raise FileNotFoundError(f"Could not find {kind} for year {year} in {year_dir}")
    cands.sort(key=lambda p: len(p.name))
    return cands[0]


def read_singleband(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
    return arr, meta


def read_p25_rgb_u8_and_b11(p25_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    P25 order: [B2, B3, B4, B8, B11, B12]
    True color RGB = [B4,B3,B2], plus B11 (SWIR).
    """
    with rasterio.open(p25_path) as src:
        meta = src.meta.copy()
        b2 = src.read(1).astype(np.float32)
        b3 = src.read(2).astype(np.float32)
        b4 = src.read(3).astype(np.float32)
        b11 = src.read(5).astype(np.float32)
    rgb = np.stack([to_u8_reflectance(b4), to_u8_reflectance(b3), to_u8_reflectance(b2)], axis=-1)
    return rgb, b11, meta


def auto_thresholds(ndsi: np.ndarray, ndvi: np.ndarray, b11: np.ndarray, ice_mask: np.ndarray):
    """Derive thresholds from ICE_MASK pixels (fallback to whole image if too few)."""
    ref = ice_mask.astype(bool)
    if ref.sum() < 1000:
        ref = np.isfinite(ndsi)

    ndsi_ref = ndsi[ref]
    ndvi_ref = ndvi[ref]
    b11_ref  = b11[ref]

    ndsi_thr = float(np.percentile(ndsi_ref, NDSI_BASE_P) - NDSI_MARGIN)
    b11_max  = float(np.percentile(b11_ref,  B11_BASE_P)  + B11_MARGIN)
    ndvi_max = float(np.percentile(ndvi_ref, NDVI_BASE_P) + NDVI_MARGIN)

    ndsi_thr = float(clamp(ndsi_thr, NDSI_MIN, NDSI_MAX))
    b11_max  = float(clamp(b11_max,  B11_MIN,  B11_MAX))
    ndvi_max = float(clamp(ndvi_max, NDVI_MIN, NDVI_MAX))

    return ndsi_thr, ndvi_max, b11_max


# =========================
# Main per year
# =========================
def run_year(year: str):
    year_dir = INPUTS_DIR / year
    boxes_path = PROMPTS_DIR / f"{year}_boxes.geojson"
    points_path = PROMPTS_DIR / f"{year}_points.geojson"

    if not year_dir.exists() or not boxes_path.exists() or not points_path.exists():
        print(f"[WARN] {year}: missing inputs/prompts, skipping")
        return

    p25_path  = find_year_file(year_dir, year, "P25")
    ndsi_path = find_year_file(year_dir, year, "NDSI")
    ndvi_path = find_year_file(year_dir, year, "NDVI")
    ice_path  = find_year_file(year_dir, year, "ICE_MASK")

    rgb_u8, b11, meta = read_p25_rgb_u8_and_b11(p25_path)
    ndsi, _ = read_singleband(ndsi_path)
    ndvi, _ = read_singleband(ndvi_path)
    ice, _  = read_singleband(ice_path)

    ndsi = ndsi.astype(np.float32)
    ndvi = ndvi.astype(np.float32)
    ice_mask = (ice > 0)

    height, width = ndsi.shape
    transform = meta["transform"]

    # Auto thresholds per year
    ndsi_thr, ndvi_max, b11_max = auto_thresholds(ndsi, ndvi, b11, ice_mask)
    print(f"[AUTO] {year}: NDSI>{ndsi_thr:.3f}, NDVI<{ndvi_max:.3f}, B11<{b11_max:.1f} (from ICE_MASK stats)")

    # Prompts
    boxes = gpd.read_file(boxes_path)
    pts = gpd.read_file(points_path)
    if boxes.crs is None:
        boxes = boxes.set_crs("EPSG:25833")
    if pts.crs is None:
        pts = pts.set_crs("EPSG:25833")
    pts_by_blob = {int(k): v for k, v in pts.groupby("blob_id")} if "blob_id" in pts.columns else {}

    # Predictor
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH))
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb_u8)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    rgb_valid = (rgb_u8[..., 0] > 0) | (rgb_u8[..., 1] > 0) | (rgb_u8[..., 2] > 0)

    used = 0
    skipped_low_valid = 0
    nonempty_after_gate = 0

    for _, row in tqdm(boxes.iterrows(), total=len(boxes), desc=f"SAM+auto-gate {year}"):
        blob_id = int(row["blob_id"])
        pix_box = map_bbox_to_pixel_box(transform, row.geometry.bounds, width, height)
        x0, y0, x1, y1 = map(int, pix_box.tolist())

        # Skip if almost no candidate ice in this box
        box_valid = ice_mask[y0:y1+1, x0:x1+1]
        valid_frac = float(box_valid.mean()) if box_valid.size else 0.0
        if valid_frac < MIN_VALID_FRAC_IN_BOX:
            skipped_low_valid += 1
            continue

        grp = pts_by_blob.get(blob_id)
        point_coords = None
        point_labels = None
        if grp is not None and len(grp) > 0:
            point_coords, point_labels = map_points_to_pixel(transform, grp)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=pix_box,
            multimask_output=False
        )

        used += 1
        m = masks[0].astype(np.uint8)

        # Gate (auto thresholds)
        gate = (ndsi > ndsi_thr) & (ndvi < ndvi_max) & (b11 < b11_max) & rgb_valid
        if USE_ICE_MASK:
            gate = gate & ice_mask

        m = (m.astype(bool) & gate).astype(np.uint8)

        if m.sum() > 0:
            nonempty_after_gate += 1

        full_mask = np.maximum(full_mask, m)

    # Raster morphology cleanup
    if DO_MORPHOLOGY:
        full_bool = full_mask.astype(bool)
        full_bool = remove_small_objects(full_bool, min_size=REMOVE_SMALL_OBJECTS_MIN_PIX)
        full_bool = remove_small_holes(full_bool, area_threshold=REMOVE_SMALL_HOLES_MAX_PIX)
        full_mask = full_bool.astype(np.uint8)

    ones = int(full_mask.sum())
    print(f"[DEBUG] {year}: used={used}, nonempty_after_gate={nonempty_after_gate}, skipped_low_valid={skipped_low_valid}")
    print(f"[INFO]  {year}: mask ones={ones} ({100*ones/full_mask.size:.6f}% pixels)")

    # Write mask GeoTIFF
    out_year_dir = MASKS_DIR / year
    out_year_dir.mkdir(parents=True, exist_ok=True)
    out_mask = out_year_dir / f"glacier_mask_{year}.tif"

    meta_out = meta.copy()
    meta_out.update(count=1, dtype="uint8", compress="lzw", nodata=0)
    with rasterio.open(out_mask, "w", **meta_out) as dst:
        dst.write(full_mask, 1)
    print(f"[OK] {year}: wrote mask -> {out_mask}")

    # Polygonize -> SHP (raw)
    gdf = polygonize_mask(full_mask, meta, year)

    out_shp_dir = VECTORS_DIR / f"glaciers_{year}_shp"
    out_shp_dir.mkdir(parents=True, exist_ok=True)

    out_shp_raw = out_shp_dir / f"glaciers_{year}.shp"
    gdf.to_file(out_shp_raw, driver="ESRI Shapefile")
    print(f"[OK] {year}: wrote polygons -> {out_shp_raw} ({len(gdf)} features)")

    # Smoothed vector output
    if SMOOTH_OUTPUT and len(gdf) > 0:
        gdf_s = smooth_gdf(gdf, SMOOTH_BUFFER_M, SIMPLIFY_M)
        out_shp_s = out_shp_dir / f"glaciers_{year}_smoothed.shp"
        gdf_s.to_file(out_shp_s, driver="ESRI Shapefile")
        print(f"[OK] {year}: wrote SMOOTHED polygons -> {out_shp_s} ({len(gdf_s)} features)\n")
    else:
        print()


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {CHECKPOINT_PATH}")

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    years = sorted([p.name for p in INPUTS_DIR.iterdir() if p.is_dir() and p.name.isdigit()])
    if ONLY_YEARS is not None:
        years = [y for y in years if y in ONLY_YEARS]

    print("Years:", years)
    for y in years:
        run_year(y)


if __name__ == "__main__":
    main()

