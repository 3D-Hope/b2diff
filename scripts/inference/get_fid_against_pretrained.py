"""
FID Score Calculator
Usage: python calc_fid.py <dir2> [dir1]
"""

import argparse
from cleanfid import fid

DEFAULT_DIR1 = "outputs/infer_in_cluster_template_1_pretrained/stage0/images/"


def compute_fid(dir1: str, dir2: str, clip=False) -> float:
    if clip:
        score = fid.compute_fid(dir1, dir2, mode="clean", model_name="clip_vit_b_32")
        return score

    score = fid.compute_fid(dir1, dir2)
    return score


def main():
    parser = argparse.ArgumentParser(description="Compute FID score between two image directories.")
    parser.add_argument("dir2", type=str, help="Path to second image directory.")
    parser.add_argument(
        "--dir1",
        type=str,
        default=DEFAULT_DIR1,
        help=f"Path to first image directory (default: {DEFAULT_DIR1})",
    )
    args = parser.parse_args()

    print(f"Computing FID between:\n  dir1: {args.dir1}\n  dir2: {args.dir2}\n")
    score = compute_fid(args.dir1, args.dir2, clip=True)
    print(f"FID Score (CLIP): {score:.4f}")


if __name__ == "__main__":
    main()


"""
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/vanilla_ddpo/stage21
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/b2diffu_try2/stage23
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/incremental_branch_lambda_2_fk_4particles/stage32
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/new_incremental_4_8_12_16_only_lambda_2_fk_4particles/stage35
"""

# python3 ./scripts/inference/get_fid_against_pretrained.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/vanilla_ddpo/stage21
