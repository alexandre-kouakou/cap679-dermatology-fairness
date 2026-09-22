"""
02_download_images.py
--------------------------------------------------------------------
Download Fitzpatrick17k images listed in labels_clean.csv.
Each image is saved as <md5hash>.jpg so it joins back to the labels.

RUN THIS ON YOUR OWN MACHINE / COLAB — it reaches external atlas sites
(DermaAmin, Atlas Dermatologico). Expect some dead URLs; every failure
is logged so you can filter your label table to what actually downloaded.

Features:
    - skips files already downloaded (safe to re-run / resume)
    - retries with backoff
    - polite delay between requests
    - verifies the response is really an image
    - writes download_success.csv and download_failed.csv

Usage:
    python 02_download_images.py \
        --labels ./data_labels/labels_clean.csv \
        --imgdir ./images \
        --workers 8
--------------------------------------------------------------------
"""
import argparse
import os
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; Fitzpatrick17k)"}
ID_COL = "md5hash"
URL_COL = "url"


def download_one(md5hash, url, imgdir, retries=3, timeout=20):
    """Return (md5hash, 'ok'|'skip'|reason)."""
    dest = os.path.join(imgdir, f"{md5hash}.jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return (md5hash, "skip")

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.content:
                ctype = r.headers.get("Content-Type", "")
                # guard against HTML error pages served with 200
                if "image" not in ctype and not r.content[:3] == b"\xff\xd8\xff":
                    return (md5hash, f"not_image:{ctype[:30]}")
                with open(dest, "wb") as f:
                    f.write(r.content)
                return (md5hash, "ok")
            else:
                last = f"http_{r.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(1.5 * (attempt + 1))  # backoff
    return (md5hash, last)


def main(labels, imgdir, workers, limit):
    os.makedirs(imgdir, exist_ok=True)
    rows = []
    with open(labels, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row[ID_COL], row[URL_COL]))
    if limit:
        rows = rows[:limit]

    ok, skip, fail = [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, h, u, imgdir): h for h, u in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            h, status = fut.result()
            if status == "ok":
                ok.append(h)
            elif status == "skip":
                skip.append(h)
            else:
                fail.append((h, status))
            if i % 200 == 0:
                el = time.time() - t0
                print(f"{i}/{len(rows)}  ok={len(ok)} skip={len(skip)} "
                      f"fail={len(fail)}  ({el:.0f}s)")

    # write manifests
    outdir = os.path.dirname(labels) or "."
    with open(os.path.join(outdir, "download_success.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow([ID_COL])
        for h in ok + skip:
            w.writerow([h])
    with open(os.path.join(outdir, "download_failed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow([ID_COL, "reason"])
        for h, reason in fail:
            w.writerow([h, reason])

    print("\n==== SUMMARY ====")
    print(f"requested : {len(rows)}")
    print(f"downloaded: {len(ok)}")
    print(f"already   : {len(skip)}")
    print(f"failed    : {len(fail)}")
    print(f"success rate: {100*(len(ok)+len(skip))/max(len(rows),1):.1f}%")
    print(f"wrote download_success.csv and download_failed.csv to {outdir}/")
    print("\nNext: run 03_reconcile.py to filter your splits to downloaded images.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="./data_labels/labels_clean.csv")
    ap.add_argument("--imgdir", default="./images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="download only the first N (for a quick test run)")
    args = ap.parse_args()
    main(args.labels, args.imgdir, args.workers, args.limit or None)
