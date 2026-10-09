"""Verify that the local source file is the copy this project was built from.

Source Library's "Download" button has no stable URL, so the file is
downloaded by hand (see README). This script checks that it is present,
complete, and has the expected content checksum.

Run:  python -m scripts.verify_source
"""
from src import config
from src.ingest import PAGE_MARKER_RE, content_sha256


def main() -> int:
    path = config.SOURCE_PATH
    relative = path.relative_to(config.PROJECT_ROOT)
    if not path.exists():
        print(f"Missing source file: {relative}")
        print(f"  1. Open {config.SOURCE_URL}")
        print("  2. Use Download to save the plain-text (.txt) version.")
        print(f"  3. Save it as {relative}, then run this script again.")
        return 1

    raw = path.read_bytes()
    digest = content_sha256(raw)
    pages = len(PAGE_MARKER_RE.findall(raw.decode("utf-8")))

    print(f"File            : {relative} ({len(raw):,} bytes)")
    print(f"Pages found     : {pages} (expected {config.SOURCE_PAGE_COUNT})")
    print(f"Content SHA-256 : {digest}")
    print(f"Expected        : {config.SOURCE_CONTENT_SHA256}")

    if pages == config.SOURCE_PAGE_COUNT and digest == config.SOURCE_CONTENT_SHA256:
        print("OK: source matches the verified copy.")
        return 0
    print("MISMATCH: this is not the copy the project was verified against.")
    print("Source Library may have updated its transcription. Review the changes")
    print("before updating SOURCE_CONTENT_SHA256 in src/config.py.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
