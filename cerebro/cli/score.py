"""CLI command for scoring submission.zip locally with R5 data."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_scores(output: str) -> dict:
    """Parse NRMSE scores from local_scoring.py output.

    Args:
        output: Console output from local_scoring.py

    Returns:
        Dictionary with challenge1, challenge2, and overall scores
    """
    scores = {}

    # Parse Challenge 1 NRMSE
    match = re.search(r"Challenge 1 Scores:.*?NRMSE: ([\d.]+)", output, re.DOTALL)
    if match:
        scores["challenge1"] = float(match.group(1))

    # Parse Challenge 2 NRMSE
    match = re.search(r"Challenge 2 Scores:.*?NRMSE: ([\d.]+)", output, re.DOTALL)
    if match:
        scores["challenge2"] = float(match.group(1))

    # Parse Overall Score
    match = re.search(r"Overall Score:.*?: ([\d.]+)", output, re.DOTALL)
    if match:
        scores["overall"] = float(match.group(1))

    return scores


def display_scores(scores: dict, submission_path: str):
    """Display scores in a formatted table.

    Args:
        scores: Dictionary with challenge1, challenge2, and overall scores
        submission_path: Path to submission.zip
    """
    print("\n" + "="*60)
    print(f"SUBMISSION EVALUATION: {submission_path}")
    print("="*60)

    if "challenge1" in scores:
        print(f"\n📊 Challenge 1 (Response Time Prediction)")
        print(f"   NRMSE: {scores['challenge1']:.4f}")
        print(f"   Weight: 30%")

    if "challenge2" in scores:
        print(f"\n📊 Challenge 2 (Externalizing Score Prediction)")
        print(f"   NRMSE: {scores['challenge2']:.4f}")
        print(f"   Weight: 70%")

    if "overall" in scores:
        print(f"\n🏆 Overall Score (Lower is Better)")
        print(f"   NRMSE: {scores['overall']:.4f}")
        print(f"   Formula: 0.3 × C1 + 0.7 × C2")

    print("\n" + "="*60)


def score_cli():
    """CLI entry point for cerebro-score command."""
    # Load environment variables from .env file
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Score submission.zip locally using R5 validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score with automatic data directory detection
  cerebro-score submission.zip

  # Specify data directory explicitly
  cerebro-score submission.zip --data-dir data

  # Fast dev run (single subject)
  cerebro-score submission.zip --fast-dev-run

  # Keep extraction directory for inspection
  cerebro-score submission.zip --keep-output
        """
    )

    parser.add_argument(
        "submission_zip",
        type=str,
        help="Path to submission.zip"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing HBN releases (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for extraction and predictions (default: temporary directory)"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Fast dev run: only evaluate one subject (for testing)"
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep output directory after scoring (default: clean up)"
    )

    args = parser.parse_args()

    # Validate submission exists
    submission_path = Path(args.submission_zip)
    if not submission_path.exists():
        print(f"Error: Submission not found: {submission_path}", file=sys.stderr)
        return 1

    # Validate data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        print("Expected to contain HBN releases (R1-R11)", file=sys.stderr)
        return 1

    # Create or use output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup = args.keep_output  # Only cleanup if user didn't specify --keep-output
    else:
        # Use temporary directory
        output_dir = Path(tempfile.mkdtemp(prefix="cerebro_score_"))
        cleanup = not args.keep_output  # Cleanup unless --keep-output specified

    try:
        # Construct command
        cmd = [
            "uv", "run", "python",
            "startkit/local_scoring.py",
            "--submission-zip", str(submission_path),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
        ]

        if args.fast_dev_run:
            cmd.append("--fast-dev-run")

        # Run scoring
        print(f"\n🔄 Running local scoring...")
        print(f"   Submission: {submission_path}")
        print(f"   Data: {data_dir}")
        print(f"   Output: {output_dir}")
        if args.fast_dev_run:
            print(f"   Mode: Fast dev run (single subject)")
        print()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # Display full output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Check for errors
        if result.returncode != 0:
            print(f"\n❌ Scoring failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

        # Parse and display scores
        scores = parse_scores(result.stdout)
        if scores:
            display_scores(scores, str(submission_path))
        else:
            print("⚠️  Warning: Could not parse scores from output", file=sys.stderr)

        # Show predictions file location
        predictions_file = output_dir / "predictions.pickle"
        if predictions_file.exists():
            print(f"\n💾 Predictions saved: {predictions_file}")

        if not cleanup:
            print(f"\n📁 Output directory preserved: {output_dir}")

        return 0

    except Exception as e:
        print(f"Error during scoring: {e}", file=sys.stderr)
        return 1

    finally:
        # Cleanup temporary directory if requested
        if cleanup and output_dir.exists() and output_dir.name.startswith("cerebro_score_"):
            import shutil
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    sys.exit(score_cli())
