#!/usr/bin/env python3
"""
SfM pipeline using hloc with ALIKED + LightGlue.

Matching strategy: sequential pairs + NetVLAD retrieval pairs (union).
Output: COLMAP-format sparse reconstruction.

Usage:
    python hloc_sfm.py --image_dir /path/to/images --output_dir /path/to/output
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set, Tuple

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
)

logger = logging.getLogger(__name__)


# ── configs ──────────────────────────────────────────────────────────────
FEATURE_CONF = extract_features.confs["aliked-n16"]
MATCHER_CONF = match_features.confs["aliked+lightglue"]
RETRIEVAL_CONF = extract_features.confs["netvlad"]


# ── sequential pairs ────────────────────────────────────────────────────
def generate_sequential_pairs(
    image_dir: Path,
    output: Path,
    overlap: int = 10,
    image_list: Optional[List[str]] = None,
) -> Path:
    """Write sequential pairs file: each image paired with its *overlap*
    nearest neighbours in sorted filename order."""
    if image_list is not None:
        names = sorted(image_list)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        names = sorted(
            p.relative_to(image_dir).as_posix()
            for p in image_dir.rglob("*")
            if p.suffix.lower() in exts
        )

    pairs: Set[Tuple[str, str]] = set()
    for i, name_i in enumerate(names):
        for j in range(i + 1, min(i + 1 + overlap, len(names))):
            pairs.add((name_i, names[j]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for n1, n2 in sorted(pairs):
            f.write(f"{n1} {n2}\n")
    logger.info("Sequential pairs: %d written to %s", len(pairs), output)
    return output


# ── merge pairs ─────────────────────────────────────────────────────────
def merge_pairs(pair_files: List[Path], output: Path) -> Path:
    """Union-merge multiple pair files into one, removing duplicates."""
    pairs: Set[Tuple[str, str]] = set()
    for pf in pair_files:
        with open(pf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    a, b = parts
                    # normalise order so (a,b) == (b,a)
                    pairs.add((min(a, b), max(a, b)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for n1, n2 in sorted(pairs):
            f.write(f"{n1} {n2}\n")
    logger.info("Merged pairs: %d unique written to %s", len(pairs), output)
    return output


# ── main pipeline ───────────────────────────────────────────────────────
def run_sfm(
    image_dir: Path,
    output_dir: Path,
    seq_overlap: int = 10,
    retrieval_num: int = 20,
    image_list: Optional[List[str]] = None,
) -> None:
    """Run the full SfM pipeline.

    Steps
    -----
    1. Extract ALIKED-N16 local features.
    2. Extract NetVLAD global descriptors.
    3. Build retrieval pairs (top-k by NetVLAD similarity).
    4. Build sequential pairs (sorted filename order, ±overlap).
    5. Merge (union) both pair sets.
    6. Match with LightGlue on the merged pairs.
    7. Triangulate / reconstruct with COLMAP (via pycolmap).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sfm_dir = output_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    # paths
    features_path = output_dir / "features.h5"
    retrieval_path = output_dir / "global_features.h5"
    pairs_seq = output_dir / "pairs-seq.txt"
    pairs_ret = output_dir / "pairs-retrieval.txt"
    pairs_merged = output_dir / "pairs-merged.txt"
    matches_path = output_dir / "matches.h5"

    # 1. local features
    logger.info("=== Step 1: Extract ALIKED-N16 local features ===")
    extract_features.main(
        FEATURE_CONF, image_dir, feature_path=features_path, image_list=image_list
    )

    # 2. global features for retrieval
    logger.info("=== Step 2: Extract NetVLAD global descriptors ===")
    extract_features.main(
        RETRIEVAL_CONF, image_dir, feature_path=retrieval_path, image_list=image_list
    )

    # 3. retrieval pairs
    logger.info("=== Step 3: Build retrieval pairs (top-%d) ===", retrieval_num)
    pairs_from_retrieval.main(
        retrieval_path, pairs_ret, num_matched=retrieval_num
    )

    # 4. sequential pairs
    logger.info("=== Step 4: Build sequential pairs (overlap=%d) ===", seq_overlap)
    generate_sequential_pairs(
        image_dir, pairs_seq, overlap=seq_overlap, image_list=image_list
    )

    # 5. merge
    logger.info("=== Step 5: Merge pairs ===")
    merge_pairs([pairs_seq, pairs_ret], pairs_merged)

    # 6. match
    logger.info("=== Step 6: Match with ALIKED + LightGlue ===")
    match_features.main(
        MATCHER_CONF,
        pairs_merged,
        features=features_path,
        matches=matches_path,
    )

    # 7. reconstruct
    logger.info("=== Step 7: COLMAP reconstruction ===")
    model = reconstruction.main(
        sfm_dir,
        image_dir,
        pairs_merged,
        features_path,
        matches_path,
        image_list=image_list,
    )

    stats = model.summary()
    logger.info("Reconstruction done.  %s", stats)
    logger.info("COLMAP model written to: %s", sfm_dir)


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SfM with hloc: ALIKED + LightGlue, "
        "sequential + retrieval matching"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write all outputs (features, matches, sfm model)",
    )
    parser.add_argument(
        "--seq_overlap",
        type=int,
        default=10,
        help="Number of sequential neighbours to pair (default: 10)",
    )
    parser.add_argument(
        "--retrieval_num",
        type=int,
        default=20,
        help="Number of top retrieval matches per image (default: 20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    run_sfm(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        seq_overlap=args.seq_overlap,
        retrieval_num=args.retrieval_num,
    )
