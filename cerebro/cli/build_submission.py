"""CLI command for building submission.zip from Lightning checkpoints."""

import argparse
import sys
from pathlib import Path

from cerebro.submission import SubmissionBuilder


def build_submission_cli():
    """CLI entry point for cerebro build-submission command."""
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Build submission.zip from Lightning checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from single checkpoint (reuse for both challenges)
  cerebro build-submission --checkpoint outputs/best.ckpt

  # Build from separate checkpoints
  cerebro build-submission \\
      --challenge1-ckpt outputs/challenge1/best.ckpt \\
      --challenge2-ckpt outputs/challenge2/best.ckpt

  # Specify output path
  cerebro build-submission \\
      --checkpoint outputs/best.ckpt \\
      --output submission_v1.zip
        """
    )

    # Checkpoint arguments (mutually exclusive with checkpoint)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint to use for both challenges (default: reuse for both)"
    )
    parser.add_argument(
        "--challenge1-ckpt",
        type=str,
        help="Challenge 1 checkpoint path"
    )
    parser.add_argument(
        "--challenge2-ckpt",
        type=str,
        help="Challenge 2 checkpoint path"
    )

    # Output path
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="submission.zip",
        help="Output path for submission.zip (default: submission.zip)"
    )

    args = parser.parse_args()

    # Validate checkpoint arguments
    if args.checkpoint:
        if args.challenge1_ckpt or args.challenge2_ckpt:
            parser.error("Cannot use --checkpoint with --challenge1-ckpt or --challenge2-ckpt")
        challenge1_ckpt = args.checkpoint
        challenge2_ckpt = args.checkpoint
    else:
        if not args.challenge1_ckpt or not args.challenge2_ckpt:
            parser.error("Must provide either --checkpoint OR both --challenge1-ckpt and --challenge2-ckpt")
        challenge1_ckpt = args.challenge1_ckpt
        challenge2_ckpt = args.challenge2_ckpt

    # Build submission
    try:
        builder = SubmissionBuilder(
            challenge1_ckpt=challenge1_ckpt,
            challenge2_ckpt=challenge2_ckpt
        )
        output_path = builder.build(args.output)
        print(f"\n✓ Successfully created {output_path}")
        print("\nNext steps:")
        print(f"  1. Test locally: cerebro score {output_path}")
        print("  2. Submit to Codabench: https://www.codabench.org/competitions/4353/")
        return 0

    except Exception as e:
        print(f"Error building submission: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(build_submission_cli())
