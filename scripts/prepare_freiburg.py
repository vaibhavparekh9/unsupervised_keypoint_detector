"""Extract Freiburg Static Cars 52 into data/freiburg_cars/.

Expects data/downloads/freiburg_static_cars_52_v1.1.tar.gz (see README for
the URL). Result layout: data/freiburg_cars/{carNNN/*.png, annotations/*.txt}
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARBALL = os.path.join(REPO, "data/downloads/freiburg_static_cars_52_v1.1.tar.gz")
OUT = os.path.join(REPO, "data/freiburg_cars")


def main():
    if not os.path.exists(TARBALL):
        print("MANUAL STEP REQUIRED: download the dataset first:\n"
              "  curl -L -o data/downloads/freiburg_static_cars_52_v1.1.tar.gz \\\n"
              "    https://lmb.informatik.uni-freiburg.de/resources/datasets/"
              "FreiburgStaticCars52/freiburg_static_cars_52_v1.1.tar.gz")
        sys.exit(3)
    if os.path.isdir(os.path.join(OUT, "annotations")):
        n = len([d for d in os.listdir(OUT) if d.startswith("car")])
        if n >= 50:
            print(f"already extracted: {OUT} ({n} sequences)")
            return
    os.makedirs(OUT, exist_ok=True)
    print(f"extracting {TARBALL} -> {OUT} (4.5 GB, a few minutes)...")
    subprocess.run(["tar", "xzf", TARBALL, "-C", OUT], check=True)
    # tarball has a leading ./ root
    n = len([d for d in os.listdir(OUT) if d.startswith("car")])
    print(f"done: {n} car sequences")


if __name__ == "__main__":
    main()
