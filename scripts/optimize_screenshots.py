"""Convert real browser captures to compact WebP files for GitHub (no visual fabrication)."""
from pathlib import Path
from PIL import Image

screenshots = Path(__file__).resolve().parents[1] / "docs/assets/screenshots"
for source in sorted(screenshots.glob("*.png")):
    with Image.open(source) as image:
        image.convert("RGB").save(source.with_suffix(".webp"), quality=84, method=6)
    print(f"{source.stem}: {source.with_suffix('.webp').stat().st_size:,} bytes")
