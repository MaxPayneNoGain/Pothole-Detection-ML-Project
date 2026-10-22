"""
Merges the two Kaggle pothole datasets into one folder per class and drops
exact duplicate images.

Expected layout before running (download links in the README):
    data/raw/pothole-and-plain-road-images/
    data/raw/pothole-image-dataset/

Output:
    data/merged/pothole/
    data/merged/plain/
"""

import argparse
import hashlib
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root):
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    )


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Merge and dedupe the pothole datasets.")
    parser.add_argument("--roads-dir", type=Path,
                        default=Path("data/raw/pothole-and-plain-road-images"))
    parser.add_argument("--potholes-dir", type=Path,
                        default=Path("data/raw/pothole-image-dataset"))
    parser.add_argument("--out", type=Path, default=Path("data/merged"))
    args = parser.parse_args()

    for d in (args.roads_dir, args.potholes_dir):
        if not d.is_dir():
            raise SystemExit(f"missing dataset folder: {d}")

    # its train/test split is ignored, everything gets re-split after merging
    sources = []
    for p in find_images(args.roads_dir):
        cls = p.parent.name.lower()
        if cls in ("pothole", "plain"):
            sources.append((p, cls, "virenbr"))

    for p in find_images(args.potholes_dir):
        sources.append((p, "pothole", "sachin"))

    if args.out.exists():
        shutil.rmtree(args.out)
    for cls in ("pothole", "plain"):
        (args.out / cls).mkdir(parents=True)

    # the datasets share photos under different names, and a copy in both
    # train and test would inflate the score. md5 only catches exact copies
    seen = set()
    counts = {"pothole": 0, "plain": 0}
    dupes = 0
    for path, cls, source in sources:
        h = md5(path)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        shutil.copy2(path, args.out / cls / f"{source}_{h[:12]}{path.suffix.lower()}")
        counts[cls] += 1

    print(f"pothole: {counts['pothole']}")
    print(f"plain:   {counts['plain']}")
    print(f"total:   {sum(counts.values())}")
    print(f"duplicates skipped: {dupes}")


if __name__ == "__main__":
    main()
