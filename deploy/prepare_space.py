"""
Prepare a Hugging Face Space-ready copy of this repo.

    python deploy/prepare_space.py OUTPUT_DIR

Copies every git-tracked (or new, non-ignored) file into OUTPUT_DIR and prepends
deploy/hf_space_header.yml to README.md, because a Space reads its settings
(sdk: docker, app_port) from YAML at the top of README.md. We keep that header
out of the real README so it doesn't clutter GitHub. Used by CI and by hand.
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(out: str) -> None:
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    if os.path.exists(out):
        shutil.rmtree(out)
    for rel in files:
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue  # deleted but still listed
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    with open(os.path.join(ROOT, "deploy", "hf_space_header.yml"), encoding="utf-8") as f:
        header = f.read()
    readme = os.path.join(out, "README.md")
    with open(readme, encoding="utf-8", newline="") as f:
        body = f.read()
    with open(readme, "w", encoding="utf-8", newline="") as f:
        f.write(header.rstrip() + "\n\n" + body)
    print(f"Space copy ready: {out}  ({len(files)} files)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
