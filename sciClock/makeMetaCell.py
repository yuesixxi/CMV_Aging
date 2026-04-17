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
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
sc._settings.ScanpyConfig.n_jobs = -1

#input file is from the output file of datapreprocess.py
INPUT_FILE = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/external_adata_all_merge_9358.h5ad"
OUTPUT_FILE = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/EX_pseudo_adata_all_merge_9358.h5ad"

DEFAULT_CELLTYPES = ["CD4T", "CD8T", "NK", "B", "MONO"]


def make_pseudocells_for_group(
    adata: AnnData,
    group_key: str,
    group_value,
    target_celltype: str,
    n_per_pool: int = 30,
    n_pools_per_donor: int = 50,
    agg: str = "mean",
    log_base: str = "e",
    seed: int = 42,
) -> list[AnnData]:
    rng = np.random.default_rng(seed)

    if pd.isna(group_value):
        adata_sub = adata[
            (adata.obs[group_key].isna()) & (adata.obs["Major_CT"] == target_celltype)
        ].copy()
    else:
        adata_sub = adata[
            (adata.obs[group_key] == group_value) & (adata.obs["Major_CT"] == target_celltype)
        ].copy()

    if adata_sub.n_obs == 0:
        return []

    X = adata_sub.X.toarray() if issparse(adata_sub.X) else adata_sub.X
    obs = adata_sub.obs.copy()
    var = adata_sub.var.copy()

    donor_fixed_fields = [
        "orig.ident",
        "age",
        "sex",
        "Major_CT",
        "CMV",
        "ts",
        "POP",
        "Condition",
        "cohort",
    ]

    pooled_rows = []
    pooled_obs = []

    donors = obs["donor_id"].unique()

    for donor in donors:
        donor_mask = obs["donor_id"].values == donor
        donor_idx = np.where(donor_mask)[0]

        if len(donor_idx) < n_per_pool:
            continue

        donor_values = obs.loc[donor_mask].iloc[0]

        for i in range(n_pools_per_donor):
            chosen = rng.choice(donor_idx, size=n_per_pool, replace=False)
            X_chosen = X[chosen]

            if agg == "mean":
                pooled = X_chosen.mean(axis=0)
            elif agg == "sum":
                pooled = X_chosen.sum(axis=0)
            else:
                raise ValueError("agg must be either 'mean' or 'sum'.")

            if log_base == "e":
                pooled_log = np.log1p(pooled)
            elif log_base == 2:
                pooled_log = np.log2(pooled + 1)
            elif log_base == 10:
                pooled_log = np.log10(pooled + 1)
            else:
                raise ValueError("Unsupported log_base. Use 'e', 2, or 10.")

            meta = {
                "donor_id": donor,
                "celltype": target_celltype,
                "group_key": group_key,
                "group_value": str(group_value) if not pd.isna(group_value) else "NaN",
                "n_cells_pooled": n_per_pool,
                "pool_index": i,
            }

            for col in donor_fixed_fields:
                if col in obs.columns:
                    meta[col] = donor_values[col]

            pooled_rows.append(pooled_log)
            pooled_obs.append(meta)

    if len(pooled_rows) == 0:
        return []

    X_pseudo = np.vstack(pooled_rows)
    obs_df = pd.DataFrame(pooled_obs)
    obs_df.index = [
        f"{r['celltype']}_{r['donor_id']}_{r['group_key']}={r['group_value']}_pool{r['pool_index']}"
        for _, r in obs_df.iterrows()
    ]

    pseudo_adata = AnnData(X=X_pseudo, obs=obs_df, var=var)

    for col in pseudo_adata.obs.columns:
        if col in ["donor_id", "celltype", "sex", "POP", "Condition", "Major_CT", "cohort", "group_key", "group_value"]:
            pseudo_adata.obs[col] = pseudo_adata.obs[col].astype("category")
        elif col == "CMV":
            pseudo_adata.obs[col] = pd.to_numeric(pseudo_adata.obs[col], errors="coerce").astype("Int64")
        elif col == "ts":
            if pd.api.types.is_numeric_dtype(pseudo_adata.obs[col]):
                pseudo_adata.obs[col] = pseudo_adata.obs[col].astype("float32")
            else:
                pseudo_adata.obs[col] = pseudo_adata.obs[col].fillna("unknown").astype("category")
        elif col == "age":
            pseudo_adata.obs[col] = pd.to_numeric(pseudo_adata.obs[col], errors="coerce").astype("float32")
        elif pseudo_adata.obs[col].dtype == "object":
            pseudo_adata.obs[col] = pseudo_adata.obs[col].fillna("unknown").astype("category")

    return [pseudo_adata]


def make_pseudocells_condition_ts(
    adata: AnnData,
    celltypes: list[str] | None = None,
    n_per_pool: int = 30,
    n_pools_per_donor: int = 50,
    agg: str = "mean",
    log_base: str = "e",
    seed: int = 42,
    group_keys=None,
) -> AnnData:
    if celltypes is None:
        celltypes = DEFAULT_CELLTYPES

    adata = adata[adata.obs["Major_CT"].isin(celltypes)].copy()

    if group_keys is None:
        candidates = ["Condition", "ts", "POP", "CMV", "cohort"]
        group_keys = [k for k in candidates if k in adata.obs.columns]

    used_temp_col = False
    if len(group_keys) == 0:
        adata.obs["_ALL_"] = "ALL"
        group_keys = ["_ALL_"]
        used_temp_col = True

    pseudo_list = []

    for ct in celltypes:
        for key in group_keys:
            all_values = adata.obs[key].unique()
            for group_value in all_values:
                pseudo_ct = make_pseudocells_for_group(
                    adata=adata,
                    group_key=key,
                    group_value=group_value,
                    target_celltype=ct,
                    n_per_pool=n_per_pool,
                    n_pools_per_donor=n_pools_per_donor,
                    agg=agg,
                    log_base=log_base,
                    seed=seed,
                )
                pseudo_list.extend(pseudo_ct)

    if used_temp_col:
        del adata.obs["_ALL_"]

    if len(pseudo_list) == 0:
        raise ValueError(
            "No pseudo-cells were generated. "
            "Possible reason: some donors have fewer than n_per_pool cells "
            "for a given cell type/group. Try lowering n_per_pool or n_pools_per_donor."
        )

    combined = pseudo_list[0].concatenate(
        *pseudo_list[1:],
        join="outer",
        batch_key="batch",
        index_unique=None,
    )
    return combined


def load_input_adata(input_file: str) -> AnnData:
    print(f"Reading input AnnData from:\n{input_file}")
    adata = sc.read_h5ad(input_file)

    if adata.layers:
        adata.layers.clear()
        gc.collect()

    print(f"Loaded adata shape: {adata.shape}")
    return adata


def main():
    adata = load_input_adata(INPUT_FILE)

    pseudo_adata_all = make_pseudocells_condition_ts(
        adata=adata,
        celltypes=DEFAULT_CELLTYPES,
        n_per_pool=30,
        n_pools_per_donor=50,
        agg="mean",
        log_base="e",
        seed=0,
        group_keys=["cohort"],
    )

    print(f"Pseudo-cell AnnData shape: {pseudo_adata_all.shape}")

    pseudo_adata_all.write_h5ad(OUTPUT_FILE, compression="gzip")
    print(f"Saved pseudo-cell AnnData to:\n{OUTPUT_FILE}")

    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
