#!/usr/bin/env python3
"""
Download the PLAsTiCC dataset (unblinded) from Zenodo record 2539456.

Queries the Zenodo API for the current file list, then downloads each file
with a progress bar, MD5 verification, and resume support. Re-running skips
files that are already present and pass their checksum.

Usage:
    python download_plasticc.py                 # download everything
    python download_plasticc.py -o ./data       # choose output dir
    python download_plasticc.py --train-only     # skip the large test light curves
    python download_plasticc.py --list           # just list files + sizes, no download

Only dependency is `requests` (pip install requests). Everything else is stdlib.
"""

import argparse
import hashlib
import os
import sys

import requests

ZENODO_RECORD = "2539456"
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
CHUNK = 1 << 20  # 1 MiB


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def get_file_list():
    """Return [(filename, url, size, md5), ...] from the Zenodo API."""
    r = requests.get(API_URL, timeout=60)
    r.raise_for_status()
    files = []
    for f in r.json().get("files", []):
        # Zenodo schema varies slightly; handle both old and new shapes.
        name = f.get("key") or f.get("filename")
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        size = f.get("size") or f.get("filesize") or 0
        checksum = f.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if ":" in checksum else checksum
        files.append((name, url, size, md5))
    return files


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def already_ok(path, size, md5):
    if not os.path.exists(path):
        return False
    if size and os.path.getsize(path) != size:
        return False
    if md5:
        return md5sum(path) == md5
    return True  # no checksum available; trust size match


def download(name, url, size, md5, outdir):
    dest = os.path.join(outdir, name)
    if already_ok(dest, size, md5):
        print(f"  [skip] {name} already present and verified")
        return

    # Resume support via HTTP Range.
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {}
    mode = "wb"
    if 0 < existing < (size or float("inf")):
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"  [resume] {name} from {human(existing)}")
    else:
        existing = 0

    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        r.raise_for_status()
        done = existing
        with open(dest, mode) as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if size:
                    pct = 100 * done / size
                    sys.stdout.write(
                        f"\r  {name}: {human(done)}/{human(size)} ({pct:5.1f}%)"
                    )
                else:
                    sys.stdout.write(f"\r  {name}: {human(done)}")
                sys.stdout.flush()
    print()

    if md5 and md5sum(dest) != md5:
        print(f"  [WARN] checksum mismatch for {name} — re-run to retry")
    else:
        print(f"  [ok] {name}")


def main():
    ap = argparse.ArgumentParser(description="Download PLAsTiCC data from Zenodo.")
    ap.add_argument("-o", "--outdir", default="plasticc_data", help="output directory")
    ap.add_argument("--train-only", action="store_true",
                    help="skip the large test light-curve files")
    ap.add_argument("--list", action="store_true",
                    help="list files and sizes, then exit")
    args = ap.parse_args()

    print(f"Fetching file list from Zenodo record {ZENODO_RECORD}...")
    files = get_file_list()

    if args.train_only:
        files = [f for f in files if "test_lightcurves" not in f[0].lower()]

    total = sum(f[2] for f in files)
    print(f"\n{len(files)} file(s), {human(total)} total:\n")
    for name, _, size, _ in files:
        print(f"  {human(size):>10}  {name}")
    print()

    if args.list:
        return

    os.makedirs(args.outdir, exist_ok=True)
    for name, url, size, md5 in files:
        download(name, url, size, md5, args.outdir)

    print(f"\nDone. Files in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
