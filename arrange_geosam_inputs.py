# arrange_geosam_inputs.py
# Copies P25 + ICE_MASK + NDSI + NDVI into inputs/<year>/ as:
#   P25.tif, ICE_MASK.tif, NDSI.tif, NDVI.tif

from pathlib import Path
import shutil
import re

# =========================
# CONFIG — adjust if needed
# =========================
ROOT = Path.home() / "munte/GEOSAM_Svalbard"

SOURCE_DIR = (
    ROOT / "Date"
    / "GeoSAM_Aug10_31_P25_Indices_Mask-20251225T150935Z-3-001"
    / "GeoSAM_Aug10_31_P25_Indices_Mask"
)

TARGET_INPUTS = ROOT / "inputs"

# Years to process (auto-detected if empty)
YEARS: list[str] = []

# =========================
# Helpers
# =========================
YEAR_RE = re.compile(r"S2_(\d{4})_Aug10_31_")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# =========================
# Main
# =========================
def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found:\n{SOURCE_DIR}")

    ensure_dir(TARGET_INPUTS)

    files = list(SOURCE_DIR.glob("*.tif"))

    # Auto-detect years if not provided
    years_found = set()
    for f in files:
        m = YEAR_RE.search(f.name)
        if m:
            years_found.add(m.group(1))

    years = YEARS if YEARS else sorted(years_found)
    print("Detected years:", years)

    for year in years:
        year_dir = TARGET_INPUTS / year
        ensure_dir(year_dir)

        # Expected files in SOURCE_DIR
        p25  = SOURCE_DIR / f"S2_{year}_Aug10_31_p25_EPSG25833.tif"
        mask = SOURCE_DIR / f"S2_{year}_Aug10_31_ICE_MASK_EPSG25833.tif"
        ndsi = SOURCE_DIR / f"S2_{year}_Aug10_31_NDSI_EPSG25833.tif"
        ndvi = SOURCE_DIR / f"S2_{year}_Aug10_31_NDVI_EPSG25833.tif"

        missing = []
        if not p25.exists():  missing.append("P25")
        if not mask.exists(): missing.append("ICE_MASK")
        if not ndsi.exists(): missing.append("NDSI")
        if not ndvi.exists(): missing.append("NDVI")

        if missing:
            print(f"[WARN] {year}: missing {', '.join(missing)} (skipping year)")
            continue

        # Copy + rename into inputs/<year>/
        shutil.copy2(p25,  year_dir / "P25.tif")
        shutil.copy2(mask, year_dir / "ICE_MASK.tif")
        shutil.copy2(ndsi, year_dir / "NDSI.tif")
        shutil.copy2(ndvi, year_dir / "NDVI.tif")

        print(f"[OK] {year}: copied P25 + ICE_MASK + NDSI + NDVI")

    print("\nGeoSAM input folder is ready.")

if __name__ == "__main__":
    main()

