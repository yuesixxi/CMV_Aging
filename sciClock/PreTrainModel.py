# singlecell age clock pipeline for data processing.
# Copyright (C) 2026
# - Yuesi Xi, Helmholtz Centre for Infection Research (HZI)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# coding: utf-8

import gc
import json
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc._settings.ScanpyConfig.n_jobs = -1


# =========================================================
# Input files
# =========================================================

# Internal pseudo-cell dataset
INTERNAL_INPUT_FILE = "/vol/projects/yxi24/AgingWithXun/PseudoCell.h5ad"

# External pseudo-cell dataset
EXTERNAL_INPUT_FILE = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/EX_pseudo_adata_all_merge_9358.h5ad"

# Internal donor split JSON files
INTERNAL_TEST_JSON = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/internal_test_donor_id.json"
INTERNAL_TRAIN_JSON = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/internal_train_donor_id.json"

# Output files for internal train/test data
INTERNAL_TEST_OUTPUT = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/adata_internal_test_9358.h5ad"
INTERNAL_TRAIN_OUTPUT = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/adata_internal_train_9358.h5ad"

# Output file for external donor split
EXTERNAL_SPLIT_JSON = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/external_train_donor_id.json"

# Optional output files for split external datasets
EXTERNAL_TRAIN_OUTPUT = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/adata_external_train_9358.h5ad"
EXTERNAL_TEST_OUTPUT = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/adata_external_test_9358.h5ad"


# =========================================================
# Helper functions
# =========================================================

def flatten(seq):
    for x in seq:
        if isinstance(x, (list, tuple, set)):
            yield from flatten(x)
        else:
            yield x


def load_donor_list_from_json(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support two JSON formats:
    # 1) {"donor_id": [...]}
    # 2) [...]
    if isinstance(data, dict) and "donor_id" in data:
        donor_list = data["donor_id"]
    else:
        donor_list = data

    donor_list = [str(x).strip() for x in flatten(donor_list) if x is not None]
    return donor_list


def subset_adata_by_donor_json(adata, json_path: str):
    donor_list = load_donor_list_from_json(json_path)
    donor_set = set(donor_list)

    donor_series = adata.obs["donor_id"].astype("string").str.strip()
    mask = donor_series.isin(donor_set)

    before = adata.n_obs
    adata_subset = adata[mask].copy()
    after = adata_subset.n_obs

    print(f"Loaded donor split from: {json_path}")
    print(f"Matched donors: {len(donor_set)}")
    print(f"Cells: {before} -> {after}")

    return adata_subset


def align_genes(adata_internal_test, adata_internal_train, pseudo_adata_all):
    # Align genes across adata_internal_test, adata_internal_train, and pseudo_adata_all.
    # Keep only the shared genes and reorder them according to pseudo_adata_all.

    for ad in [adata_internal_test, adata_internal_train, pseudo_adata_all]:
        ad.var_names = ad.var_names.astype(str)
        if ad.var_names.has_duplicates:
            ad.var_names_make_unique()

    shared = (
        set(adata_internal_test.var_names)
        & set(adata_internal_train.var_names)
        & set(pseudo_adata_all.var_names)
    )

    if not shared:
        raise ValueError("The three AnnData objects do not share any genes, cannot align.")

    shared_order = [g for g in pseudo_adata_all.var_names if g in shared]
    print(f"Number of shared genes: {len(shared_order)}")

    adata_internal_test = adata_internal_test[:, shared_order].copy()
    adata_internal_train = adata_internal_train[:, shared_order].copy()
    pseudo_adata_all = pseudo_adata_all[:, shared_order].copy()

    assert adata_internal_test.var_names.equals(pseudo_adata_all.var_names)
    assert adata_internal_train.var_names.equals(pseudo_adata_all.var_names)

    print(
        "Gene alignment finished:",
        f"test vars={adata_internal_test.n_vars},",
        f"train vars={adata_internal_train.n_vars},",
        f"external vars={pseudo_adata_all.n_vars}",
    )

    gc.collect()
    return adata_internal_test, adata_internal_train, pseudo_adata_all


def keep_obs_and_tag(ad, tag: str):
    keep_cols = [
        "donor_id",
        "celltype",
        "group_key",
        "group_value",
        "n_cells_pooled",
        "pool_index",
        "orig.ident",
        "age",
        "sex",
        "Major_CT",
        "cohort",
        "batch",
    ]

    # If the object is a view, make a copy first to avoid in-place modification issues
    if getattr(ad, "is_view", False):
        ad = ad.copy()

    # Fill missing columns with <NA>
    for c in keep_cols:
        if c not in ad.obs.columns:
            ad.obs[c] = pd.NA

    # Keep only the selected columns in the specified order
    ad.obs = ad.obs[keep_cols].copy()

    # Add a dataset type label
    ad.obs["type"] = tag
    ad.obs["type"] = ad.obs["type"].astype("category")
    return ad


def subset_by_donor(ad, ids: set[str]):
    donor_series = ad.obs["donor_id"].astype("string").str.strip()
    return ad[donor_series.isin(ids)].copy()


def split_external_by_donor(
    pseudo_adata_all,
    split_ratio: float = 0.7,#0.9
    seed: int = 0,
    json_out: str = EXTERNAL_SPLIT_JSON,
):
    # Split the external dataset at the donor level into train/test subsets.
    #
    # A small subset of external donors can be used as a small external anchor set
    # for model adaptation or auxiliary training.
    #


    np.random.seed(seed)

    donors = (
        pseudo_adata_all.obs["donor_id"]
        .astype("string")
        .dropna()
        .str.strip()
        .unique()
        .tolist()
    )

    np.random.shuffle(donors)

    n_test = int(len(donors) * split_ratio)
    test_ids = donors[:n_test]
    train_ids = donors[n_test:]

    split_dict = {"train": train_ids, "test": test_ids}

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved external donor split to: {json_out}")
    print(f"train donors={len(train_ids)}, test donors={len(test_ids)}")

    train_set = set(map(str, train_ids))
    test_set = set(map(str, test_ids))

    # Check that there is no donor overlap between train and test
    assert train_set.isdisjoint(test_set), "train and test donors overlap."

    adata_external_train = subset_by_donor(pseudo_adata_all, train_set)
    adata_external_test = subset_by_donor(pseudo_adata_all, test_set)

    print(
        "External split done:",
        f"external_train n_obs={adata_external_train.n_obs},",
        f"external_test n_obs={adata_external_test.n_obs}",
    )

    return adata_external_train, adata_external_test


# =========================================================
# Main
# =========================================================

def main():
    print("Reading internal pseudo-cell dataset...")
    adata = sc.read_h5ad(INTERNAL_INPUT_FILE)
    print(f"Internal input shape: {adata.shape}")

    print("\nReading external pseudo-cell dataset...")
    pseudo_adata_all = sc.read_h5ad(EXTERNAL_INPUT_FILE)
    print(f"External input shape: {pseudo_adata_all.shape}")

    # Internal train/test datasets can be created directly from donor split JSON files

    print("\nBuilding internal test set...")
    adata_internal_test = subset_adata_by_donor_json(adata, INTERNAL_TEST_JSON)

    print("\nBuilding internal train set...")
    adata_internal_train = subset_adata_by_donor_json(adata, INTERNAL_TRAIN_JSON)

    # Align genes across internal test / internal train / external data

    print("\nAligning genes...")
    adata_internal_test, adata_internal_train, pseudo_adata_all = align_genes(
        adata_internal_test,
        adata_internal_train,
        pseudo_adata_all,
    )

    # Keep selected obs columns and add a type label

    print("\nCleaning obs columns and adding type labels...")
    adata_internal_test = keep_obs_and_tag(adata_internal_test, "internal")
    adata_internal_train = keep_obs_and_tag(adata_internal_train, "internal")
    pseudo_adata_all = keep_obs_and_tag(pseudo_adata_all, "external")

    print("internal_test cols:", list(adata_internal_test.obs.columns))
    print("internal_train cols:", list(adata_internal_train.obs.columns))
    print("external cols:", list(pseudo_adata_all.obs.columns))

    # Save internal train/test datasets

    print("\nSaving internal train/test datasets...")
    adata_internal_test.write_h5ad(INTERNAL_TEST_OUTPUT, compression="gzip")
    adata_internal_train.write_h5ad(INTERNAL_TRAIN_OUTPUT, compression="gzip")

    print(f"Saved: {INTERNAL_TEST_OUTPUT}")
    print(f"Saved: {INTERNAL_TRAIN_OUTPUT}")

    # Split the external dataset.
    # A small subset of external donors may serve as a small external anchor set
    # for adaptation or auxiliary training.

    print("\nSplitting external dataset...")
    adata_external_train, adata_external_test = split_external_by_donor(
        pseudo_adata_all,
        split_ratio=0.7,
        seed=0,
        json_out=EXTERNAL_SPLIT_JSON,
    )

    adata_external_train.write_h5ad(EXTERNAL_TRAIN_OUTPUT, compression="gzip")
    adata_external_test.write_h5ad(EXTERNAL_TEST_OUTPUT, compression="gzip")

    print(f"Saved: {EXTERNAL_TRAIN_OUTPUT}")
    print(f"Saved: {EXTERNAL_TEST_OUTPUT}")

    gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()
