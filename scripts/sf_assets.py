#!/usr/bin/env python3
"""Download pinned public training assets, resume partials, verify every byte."""

import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".runtime/training"
MANIFEST = ROOT / "config/sf_training/assets.json"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r)


def manifest():
    rows = []
    base = fetch_json(
        "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/&maxResults=1000"
    )
    if base.get("nextPageToken"):
        raise RuntimeError("Incomplete GCS listing")
    for f in base["items"]:
        name = f["name"]
        rows.append(
            dict(
                path="pi05_base/" + name.split("checkpoints/pi05_base/")[1],
                size=int(f["size"]),
                digest=base64.b64decode(f.get("md5Hash", f["crc32c"])).hex(),
                algorithm="md5" if "md5Hash" in f else "crc32c",
                url="https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/"
                + urllib.parse.quote(name, safe="")
                + "?alt=media&generation="
                + f["generation"],
            )
        )
    repo = "VLA-Arena/VLA_Arena_L0_L_lerobot_openpi"
    rev = "8c6cdcac185c7c97582a78f8ad8c2ef3757ce2c7"
    ds = fetch_json(
        f"https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true"
    )
    for f in ds["siblings"]:
        p = f["rfilename"]
        lfs = f.get("lfs")
        rows.append(
            dict(
                path="arena_l0/" + p,
                size=f["size"],
                digest=lfs["sha256"] if lfs else f["blobId"],
                algorithm="sha256" if lfs else "git-sha1",
                url=f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{p}",
            )
        )
    rows.append(
        dict(
            path="vggt/model.pt",
            size=5026874952,
            algorithm="sha256",
            digest="d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0",
            url="https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt",
        )
    )
    MANIFEST.write_text(json.dumps({"files": rows}, indent=2) + "\n")


def digest(path, row):
    algo = row["algorithm"]
    if algo == "crc32c":
        import google_crc32c

        h = google_crc32c.Checksum()
    else:
        h = hashlib.new("sha1" if algo == "git-sha1" else algo)
    if algo == "git-sha1":
        h.update(f"blob {row['size']}\0".encode())
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    result = h.hexdigest()
    return result.decode() if isinstance(result, bytes) else result


def get(row, verify_only=False):
    p = DEST / row["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    if (
        p.exists()
        and p.stat().st_size == row["size"]
        and digest(p, row) == row["digest"]
    ):
        return row["path"]
    if verify_only:
        raise RuntimeError(f"Missing/corrupt: {p}")
    part = p.with_name(p.name + ".part")
    for attempt in range(5):
        try:
            start = part.stat().st_size if part.exists() else 0
            # A complete partial may only need its final verification/rename.
            if (
                part.exists()
                and start == row["size"]
                and digest(part, row) == row["digest"]
            ):
                part.replace(p)
                return row["path"]
            if part.exists() and start >= row["size"]:
                part.unlink()
                start = 0
            req = urllib.request.Request(
                row["url"], headers={"Range": f"bytes={start}-"} if start else {}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                if start and r.status != 206:
                    start = 0
                with part.open("ab" if start else "wb") as out:
                    for chunk in iter(lambda: r.read(8 * 1024 * 1024), b""):
                        out.write(chunk)
            if part.stat().st_size != row["size"] or digest(part, row) != row["digest"]:
                part.unlink()
                raise RuntimeError("Hash/size mismatch: " + row["path"])
            part.replace(p)
            return row["path"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-manifest", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.create_manifest:
        manifest()
    rows = json.loads(MANIFEST.read_text())["files"]
    done = 0
    errors = []
    ledger_path = DEST / "verified_files.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    def verify(row):
        target = DEST / row["path"]

        def fingerprint():
            state = target.stat()
            return {
                "size": state.st_size,
                "mtime_ns": state.st_mtime_ns,
                "digest": row["digest"],
                "algorithm": row["algorithm"],
            }

        if (
            not a.verify_only
            and target.exists()
            and ledger.get(row["path"]) == fingerprint()
        ):
            return row["path"], fingerprint()
        get(row, a.verify_only)
        return row["path"], fingerprint()

    def save_ledger():
        temporary = ledger_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, sort_keys=True) + "\n")
        temporary.replace(ledger_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(verify, r) for r in rows]
        for f in concurrent.futures.as_completed(futures):
            try:
                name, fingerprint = f.result()
                ledger[name] = fingerprint
            except Exception as exc:
                errors.append(repr(exc))
                print(f"Verification error: {exc}", flush=True)
            done += 1
            if done % 25 == 0 or done == len(rows):
                save_ledger()
            if done % 100 == 0 or done == len(rows):
                print(f"Verified {done}/{len(rows)} files", flush=True)
    if errors:
        raise RuntimeError("Asset verification failed: " + "; ".join(errors))
    record = {
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "files": len(rows),
        "bytes": sum(x["size"] for x in rows),
        "verified_at_unix": time.time(),
    }
    (DEST / "assets_verified.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
