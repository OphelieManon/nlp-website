"""Smoke test: the image generator script runs and writes JPGs.

Uses a tiny synthetic products CSV so we don't depend on network
(fonts) or the 1000-row real catalogue. The script downloads fonts the
first time, so the test only exercises the rendering pipeline if the
fonts directory is already populated; otherwise it's skipped.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FONTS_DIR = _REPO_ROOT / "app" / "static" / "fonts"


@pytest.mark.skipif(
    not (_FONTS_DIR / "Fraunces-VF.ttf").exists(),
    reason="Fonts not downloaded yet — run the script once to fetch them.",
)
def test_image_generator_produces_jpgs(tmp_path: Path):
    csv = tmp_path / "products.csv"
    pd.DataFrame([
        {"product_id": 1, "product_name": "Maybelline Volumizing Mascara",
         "category": "Makeup", "price": 18.0, "image_path": ""},
        {"product_id": 2, "product_name": "CeraVe Glow Face Wash",
         "category": "Skincare", "price": 12.0, "image_path": ""},
    ]).to_csv(csv, index=False)

    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.generate_product_images",
         "--csv", str(csv), "--out", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert (out / "1.jpg").exists()
    assert (out / "2.jpg").exists()
    # Sanity-check the files are non-empty JPGs
    assert (out / "1.jpg").stat().st_size > 1024
