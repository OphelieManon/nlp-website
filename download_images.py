import csv
import os
import time
import random
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse

PRODUCT_MAPPING = "data/product_mapping.csv"
OUTPUT_DIR = "data/images"
DELAY_RANGE = (1.5, 3.0)  # seconds between requests to avoid rate limiting

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def get_image_url(session, product_url):
    """Fetch product page and extract og:image URL."""
    resp = session.get(product_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Try og:image first
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]

    # Fallback: first <img> with a src that looks like a product image
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if any(ext in src.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            return src

    return None


def download_image(session, image_url, dest_path):
    """Download an image to dest_path."""
    resp = session.get(image_url, headers=HEADERS, timeout=15, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def image_extension(url):
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ".jpg"


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    failed = []

    with open(PRODUCT_MAPPING, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    session = requests.Session()

    for i, row in enumerate(rows, 1):
        product_id = row["productID"]
        page_url = row.get("image_url", "").strip()

        if not page_url:
            print(f"[{i}/{total}] {product_id}: no URL — skipping")
            failed.append(product_id)
            continue

        try:
            print(f"[{i}/{total}] {product_id}: fetching page...", end=" ", flush=True)
            image_url = get_image_url(session, page_url)

            if not image_url:
                print("no image found")
                failed.append(product_id)
                continue

            ext = image_extension(image_url)
            dest = os.path.join(OUTPUT_DIR, f"{product_id}{ext}")

            # Skip if already downloaded
            if os.path.exists(dest):
                print("already downloaded")
                continue

            download_image(session, image_url, dest)
            print(f"saved as {product_id}{ext}")

        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code}")
            failed.append(product_id)
        except Exception as e:
            print(f"error: {e}")
            failed.append(product_id)

        time.sleep(random.uniform(*DELAY_RANGE))

    print(f"\n--- Done: {total - len(failed)}/{total} downloaded ---")
    if failed:
        print(f"Failed product IDs ({len(failed)}):")
        for pid in failed:
            print(f"  {pid}")


if __name__ == "__main__":
    main()
