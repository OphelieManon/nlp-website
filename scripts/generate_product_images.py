"""Generate per-product display images for the catalogue.

Satisfies the Milestone 2 spec's "Cosmetic/Beauty Data" requirement:
*"in order to attract user's attention, you are required to create
some additional artificial data (e.g. images, ...) for the purpose
of display."*

Renders an 800x800 JPG per product into ``app/static/images/`` using
Pillow. Each image follows the editorial direction shipped in the CSS:

  • Category-tinted vertical gradient backdrop
  • Soft radial vignette + grain noise (paper texture)
  • Top corners: house mark on the left, item number on the right
  • Centre: brand (Fraunces Bold, ALL CAPS) above the product name
    (Fraunces Italic, wrapped)
  • Bottom: category label (Fraunces Regular, spaced caps) bracketed
    by hairline rules

Fonts are downloaded on first run from the Google Fonts GitHub repo into
``app/static/fonts/``. Both fonts and images are gitignored — graders
regenerate them locally before recording the demo (the README walks
through the step).

Usage:
  python -m scripts.generate_product_images
  python -m scripts.generate_product_images --limit 5      # smoke
  python -m scripts.generate_product_images --size 1200 --quality 90
"""

from __future__ import annotations

import argparse
import textwrap
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CSV = _REPO_ROOT / "knowledge" / "products.csv"
_DEFAULT_OUT = _REPO_ROOT / "app" / "static" / "images"
_FONTS_DIR = _REPO_ROOT / "app" / "static" / "fonts"

# Variable fonts — one file each for roman + italic.
# The script configures weight / optical-size axes per text run, so we
# only ever need to download two TTFs total.
_FONT_URLS = {
    "Fraunces-VF.ttf":        "https://raw.githubusercontent.com/google/fonts/main/ofl/fraunces/Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf",
    "Fraunces-Italic-VF.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/fraunces/Fraunces-Italic%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf",
}

# Same gradient stops as styles.css — RGB triples
_CATEGORY_GRADIENTS = {
    "Skincare":     ((216, 232, 213), (168, 196, 162)),
    "Makeup":       ((245, 213, 208), (216, 152, 144)),
    "Fragrance":    ((224, 213, 232), (180, 154, 196)),
    "Beauty Tools": ((213, 220, 232), (149, 164, 192)),
    "Haircare":     ((240, 220, 197), (196, 155, 111)),
}
_FALLBACK_GRADIENT = ((229, 220, 201), (201, 189, 164))

_INK = (26, 23, 20)
_INK_SOFT = (61, 53, 48)

# Multi-word brand prefixes we want to keep together for the BRAND label.
_MULTI_WORD_BRANDS = (
    "Estee Lauder",
    "The Ordinary",
    "Fenty Beauty",
    "Rare Beauty",
)


def _brand_of(product_name: str) -> str:
    """Return the brand portion of a product_name (first 1–2 words)."""
    for prefix in _MULTI_WORD_BRANDS:
        if product_name.lower().startswith(prefix.lower()):
            return prefix
    return product_name.split()[0]


def _ensure_fonts() -> None:
    """Download font files into _FONTS_DIR on first run; cached thereafter."""
    _FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in _FONT_URLS.items():
        path = _FONTS_DIR / name
        if path.exists() and path.stat().st_size > 1024:
            continue
        print(f"  downloading {name} ...", end="", flush=True)
        urllib.request.urlretrieve(url, path)
        print(" ok")


def _make_gradient(size: tuple[int, int], top: tuple[int, int, int],
                   bottom: tuple[int, int, int]) -> Image.Image:
    """Vertical 2-color gradient as RGB PIL Image."""
    w, h = size
    t = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1)
    top_arr = np.array(top, dtype=np.float32)
    bot_arr = np.array(bottom, dtype=np.float32)
    arr = top_arr + (bot_arr - top_arr) * t[..., None]   # (h, 1, 3)
    arr = np.broadcast_to(arr, (h, w, 3)).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _add_grain(img: Image.Image, strength: int = 5) -> Image.Image:
    """Add subtle film-grain noise."""
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _vignette(img: Image.Image, intensity: float = 0.18) -> Image.Image:
    """Soft dark vignette via radial gradient mask."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    d = d / d.max()
    mask = (1.0 - d.astype(np.float32) * intensity).clip(0, 1)
    arr = np.array(img, dtype=np.float32) * mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _render_one(
    product_id: int,
    brand: str,
    full_name: str,
    category: str,
    size: int = 800,
) -> Image.Image:
    top, bot = _CATEGORY_GRADIENTS.get(category, _FALLBACK_GRADIENT)
    img = _make_gradient((size, size), top, bot)
    img = _vignette(img, intensity=0.18)
    img = _add_grain(img, strength=5)
    draw = ImageDraw.Draw(img)

    # Scale font sizes to canvas size (relative to 800px baseline)
    s = size / 800
    vf_path = str(_FONTS_DIR / "Fraunces-VF.ttf")
    vf_italic_path = str(_FONTS_DIR / "Fraunces-Italic-VF.ttf")

    # Variable-font axes: (SOFT, WONK, opsz, wght)
    f_brand = ImageFont.truetype(vf_path, size=int(72 * s))
    f_brand.set_variation_by_axes([0, 0, 144, 700])      # bold display
    f_name = ImageFont.truetype(vf_italic_path, size=int(44 * s))
    f_name.set_variation_by_axes([50, 0, 72, 420])       # softened italic, mid weight
    f_small = ImageFont.truetype(vf_path, size=int(15 * s))
    f_small.set_variation_by_axes([0, 0, 9, 540])        # tight optical-size for small caps
    f_meta = ImageFont.truetype(vf_path, size=int(14 * s))
    f_meta.set_variation_by_axes([0, 0, 9, 500])

    pad = int(44 * s)

    # ── Top-left house mark ───────────────────────────────────────
    draw.text((pad, pad), "MAISON DE BEAUTÉ", font=f_meta, fill=_INK_SOFT)

    # ── Top-right item number ─────────────────────────────────────
    n_text = f"№ {product_id:04d}"
    n_w = draw.textlength(n_text, font=f_meta)
    draw.text((size - pad - n_w, pad), n_text, font=f_meta, fill=_INK_SOFT)

    # ── Centre block: BRAND + product name ────────────────────────
    cx = size / 2
    cy = size / 2 - int(24 * s)
    brand_upper = brand.upper()
    bw = draw.textlength(brand_upper, font=f_brand)
    # If brand is too wide, scale down by re-rendering at smaller size
    if bw > size - 2 * pad:
        f_brand = ImageFont.truetype(vf_path, size=int(54 * s))
        f_brand.set_variation_by_axes([0, 0, 144, 700])
        bw = draw.textlength(brand_upper, font=f_brand)
    draw.text((cx - bw / 2, cy - int(58 * s)), brand_upper, font=f_brand, fill=_INK)

    # Wrap the rest of the name (excluding brand prefix) onto up to 2 lines
    if full_name.lower().startswith(brand.lower()):
        rest = full_name[len(brand):].strip()
    else:
        rest = full_name
    wrapped = textwrap.wrap(rest, width=24)[:2]
    for i, line in enumerate(wrapped):
        lw = draw.textlength(line, font=f_name)
        draw.text((cx - lw / 2, cy + int(28 * s) + i * int(48 * s)),
                  line, font=f_name, fill=_INK_SOFT)

    # ── Bottom band: bracketed category ───────────────────────────
    cat_text = " ".join(list(category.upper()))   # letter-spaced
    cat_w = draw.textlength(cat_text, font=f_small)
    cat_y = size - int(58 * s)
    line_y = cat_y + int(9 * s)
    cat_x = (size - cat_w) / 2
    bracket_pad = int(14 * s)
    draw.line([(pad + int(20 * s), line_y), (cat_x - bracket_pad, line_y)],
              fill=_INK_SOFT, width=max(1, int(s)))
    draw.line([(cat_x + cat_w + bracket_pad, line_y),
               (size - pad - int(20 * s), line_y)],
              fill=_INK_SOFT, width=max(1, int(s)))
    draw.text((cat_x, cat_y), cat_text, font=f_small, fill=_INK)

    return img


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render per-product display images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                        help="Path to products.csv (default: knowledge/products.csv)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help="Output directory (default: webapp/static/images)")
    parser.add_argument("--size", type=int, default=800,
                        help="Edge size in pixels (default: 800)")
    parser.add_argument("--quality", type=int, default=82,
                        help="JPEG quality 1–95 (default: 82)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only render the first N products (0 = all)")
    args = parser.parse_args()

    print("Ensuring fonts are available...")
    _ensure_fonts()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(df)} images at {args.size}x{args.size} -> {args.out}/")
    for i, row in enumerate(df.itertuples(index=False), start=1):
        brand = _brand_of(row.product_name)
        img = _render_one(
            product_id=int(row.product_id),
            brand=brand,
            full_name=row.product_name,
            category=row.category,
            size=args.size,
        )
        img.save(args.out / f"{int(row.product_id)}.jpg",
                 format="JPEG", quality=args.quality, optimize=True)
        if i % 100 == 0 or i == len(df):
            print(f"  [{i:>4d}/{len(df)}]")

    print(f"Done. {len(df)} images saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
