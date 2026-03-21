"""Extract only the ~11K images we need from the full Zenodo zip.

Usage: uv run python extract_images.py
"""

import zipfile
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "ukiyoe"
ZIP_PATH = DATA_ROOT / "images_ukiyo-e.zip"
IMAGES_DIR = DATA_ROOT / "images"
NEEDED_FILE = DATA_ROOT / "needed_images.txt"


def main():
    needed = set(NEEDED_FILE.read_text().strip().split("\n"))
    print(f"Need {len(needed)} images")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name in needed:
                # Extract directly to images dir with flat structure
                data = zf.read(info.filename)
                (IMAGES_DIR / name).write_bytes(data)
                extracted += 1
                if extracted % 1000 == 0:
                    print(f"  Extracted {extracted}/{len(needed)}...")
            else:
                skipped += 1

    print(f"Done! Extracted: {extracted}, Skipped: {skipped}")

    missing = needed - {f.name for f in IMAGES_DIR.iterdir()}
    if missing:
        print(f"WARNING: {len(missing)} images not found in zip")
        for m in list(missing)[:10]:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
