"""Submission builder: Convert Lightning checkpoints → submission.zip for Codabench."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional
import zipfile

import torch


class SubmissionBuilder:
    """Build submission.zip from Lightning checkpoints.

    The submission must contain:
    - submission.py (Submission class with get_model_challenge_1/2 methods)
    - weights_challenge_1.pt (model state dict)
    - weights_challenge_2.pt (model state dict)

    Args:
        challenge1_ckpt: Path to Challenge 1 checkpoint
        challenge2_ckpt: Path to Challenge 2 checkpoint (optional, can reuse challenge1)

    Example:
        >>> builder = SubmissionBuilder(
        ...     challenge1_ckpt="outputs/challenge1/best.ckpt",
        ...     challenge2_ckpt="outputs/challenge2/best.ckpt"
        ... )
        >>> builder.build("submission.zip")
    """

    def __init__(
        self,
        challenge1_ckpt: str,
        challenge2_ckpt: Optional[str] = None,
    ):
        self.c1_ckpt = Path(challenge1_ckpt)
        self.c2_ckpt = Path(challenge2_ckpt) if challenge2_ckpt else self.c1_ckpt

        # Validate checkpoints exist
        if not self.c1_ckpt.exists():
            raise FileNotFoundError(f"Challenge 1 checkpoint not found: {self.c1_ckpt}")
        if not self.c2_ckpt.exists():
            raise FileNotFoundError(f"Challenge 2 checkpoint not found: {self.c2_ckpt}")

    def build(self, output_path: str = "submission.zip"):
        """Build submission.zip.

        Args:
            output_path: Output path for submission.zip

        Returns:
            Path to created submission.zip
        """
        output_path = Path(output_path)

        print(f"Building submission.zip...")
        print(f"  Challenge 1 checkpoint: {self.c1_ckpt}")
        print(f"  Challenge 2 checkpoint: {self.c2_ckpt}")

        # Create temporary directory for staging
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Extract and save model weights
            print("Extracting model weights...")
            self._extract_weights(self.c1_ckpt, tmpdir / "weights_challenge_1.pt")
            self._extract_weights(self.c2_ckpt, tmpdir / "weights_challenge_2.pt")

            # Generate submission.py from template
            print("Generating submission.py...")
            self._generate_submission_file(tmpdir / "submission.py")

            # Create zip file
            print(f"Creating {output_path}...")
            self._create_zip(tmpdir, output_path)

        print(f"✓ Created {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

        # Verify zip contents
        self._verify_zip(output_path)

        return output_path

    def _extract_weights(self, ckpt_path: Path, output_path: Path):
        """Extract model weights from Lightning checkpoint.

        Lightning checkpoint structure:
        - checkpoint['state_dict'] contains trainer state
        - Keys like 'model.encoder.features.0.weight' (RegressorModel wraps encoder)
        - We need to extract just the encoder weights (remove 'model.encoder.' prefix)
        """
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Extract encoder state dict and remove nested prefixes
        # Structure: model.encoder.full_model.block_X → block_X (EEGNeX)
        #            model.encoder.features.X → skip (internal feature extractor)
        encoder_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.encoder.'):
                # Remove 'model.encoder.' prefix (len=14)
                new_key = key[14:]

                # Skip internal feature extractors (EEGNeX has both full_model and features)
                if new_key.startswith('features.'):
                    continue

                # Remove nested 'full_model.' prefix if present (EEGNeX)
                if new_key.startswith('full_model.'):
                    new_key = new_key[11:]  # len('full_model.') = 11

                encoder_state_dict[new_key] = value

        if not encoder_state_dict:
            raise ValueError(f"No encoder weights found in checkpoint {ckpt_path}")

        # Save as standalone state dict
        torch.save(encoder_state_dict, output_path)
        print(f"    → Saved {len(encoder_state_dict)} parameters to {output_path.name}")

    def _generate_submission_file(self, output_path: Path):
        """Generate submission.py from template.

        The submission.py uses braindecode models directly (no cerebro imports)
        to keep the submission zip small.
        """
        # Import template here to get the raw code
        from . import template

        # Copy template code to submission.py
        template_path = Path(template.__file__)
        shutil.copy(template_path, output_path)
        print(f"    → Generated submission.py")

    def _create_zip(self, source_dir: Path, output_path: Path):
        """Create zip file from source directory."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in source_dir.iterdir():
                if file.is_file():
                    zipf.write(file, file.name)

    def _verify_zip(self, zip_path: Path):
        """Verify submission.zip contains required files."""
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            files = zipf.namelist()

        required_files = ['submission.py', 'weights_challenge_1.pt', 'weights_challenge_2.pt']
        missing = [f for f in required_files if f not in files]

        if missing:
            raise ValueError(f"Missing files in submission.zip: {missing}")

        print(f"✓ Verified submission.zip contains:")
        for file in files:
            print(f"    - {file}")
