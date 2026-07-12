"""Create the committed car split (thesis convention).

Sorted car ids: first 500 form the training pool, the rest are held-out test
cars. Dev smoke subsets are small prefixes for the 8 GB dev GPU.

Usage: python scripts/make_split.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_ROOT = "/home/vsparekh/3DRealCars-English"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "configs", "split.json")

NUM_TRAIN_POOL = 500
NUM_DEV_SMOKE = 20      # TOBECHANGED: lab uses train_pool directly (3090)
NUM_DEV_TEST_SMOKE = 10
NUM_TEST_EVAL = 200     # lab eval subset of test (feature-cache disk bound);
                        # the full `test` list remains for final numbers
NUM_LAB_TRAIN = 400     # lab PC only has cars 0000-0499 on disk, so its
                        # held-out test set must come out of those 500:
                        # lab_train = first 400, lab_test = last 100


def main():
    cars = sorted(
        d for d in os.listdir(IMAGE_ROOT)
        if os.path.isdir(os.path.join(IMAGE_ROOT, d)) and not d.startswith(".")
    )
    split = {
        "train_pool": cars[:NUM_TRAIN_POOL],
        "test": cars[NUM_TRAIN_POOL:],
        "dev_smoke": cars[:NUM_DEV_SMOKE],
        "dev_test_smoke": cars[NUM_TRAIN_POOL:NUM_TRAIN_POOL + NUM_DEV_TEST_SMOKE],
        "test_eval": cars[NUM_TRAIN_POOL:NUM_TRAIN_POOL + NUM_TEST_EVAL],
        "lab_train": cars[:NUM_LAB_TRAIN],
        "lab_test": cars[NUM_LAB_TRAIN:NUM_TRAIN_POOL],
    }
    with open(OUT, "w") as f:
        json.dump(split, f, indent=1)
    print(f"Wrote {OUT}: train_pool={len(split['train_pool'])} "
          f"test={len(split['test'])} dev_smoke={len(split['dev_smoke'])} "
          f"dev_test_smoke={len(split['dev_test_smoke'])}")


if __name__ == "__main__":
    main()
