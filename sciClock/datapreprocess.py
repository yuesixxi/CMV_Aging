#!/usr/bin/env python
# coding: utf-8

import gc
import re
import warnings

import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc._settings.ScanpyConfig.n_jobs = -1


DATA_PATHS = {
    "data1": "/path/to/scdata/output/processed_external_data/data1_annotated.h5ad",
    "data2": "/path/to/scdata/output/processed_external_data/data2_annotated.h5ad",
    "data3": "/path/to/scdata/output/processed_external_data/data3_annotated.h5ad",
    "data4": "/path/to/scdata/output/processed_external_data/data4_annotated.h5ad",
    "data5": "/path/to/scdata//processed_external_data/data5_annotated.h5ad",
    "data6": "/path/to/scdata//processed_external_data/data6_annotated.h5ad",
    "data7": "/path/to/scdata//processed_external_data/data7_annotated.h5ad",
    "data10": "/path/to/scdata//processed_external_data/data10_annotated.h5ad",
}

KEEP_COLS = [
    "orig.ident",
    "nCount_RNA",
    "nFeature_RNA",
    "donor_id",
    "age",
    "sex",
    "Major_CT",
]

SHARED_GENES_FILE = "/vol/projects/jxun/Yuesi_CMV_aging/shared_gens9901.txt"
OUTPUT_FILE = "/vol/projects/yxi24/AgingWithXun/ExternaTestlData/external_adata_all_merge_9358.h5ad"

ORDER_KEYS = ["data1", "data2", "data3", "data4", "data5", "data6", "data7", "data10"]
LABELS = ["adata1", "adata2", "adata3", "adata4", "adata5", "adata6", "adata7", "adata10"]


def load_datasets():
    datasets = {}
    for name, path in DATA_PATHS.items():
        print(f"Loading {name} from: {path}")
        datasets[name] = sc.read_h5ad(path)
    gc.collect()
    return datasets


def extract_age_min_max(age_series: pd.Series):
    if age_series is None or len(age_series) == 0:
        return None, None

    values = []

    ages_num = pd.to_numeric(age_series, errors="coerce")
    values.extend(ages_num.dropna().astype(float).tolist())

    mask = ages_num.isna()
    if mask.any():
        for s in age_series[mask].astype(str):
            nums = re.findall(r"\d+(?:\.\d+)?", s)
            for x in nums:
                try:
                    values.append(float(x))
                except Exception:
                    pass

    if not values:
        return None, None

    return float(min(values)), float(max(values))


def summarize_datasets(datasets):
    rows = []
    for name, ad in datasets.items():
        obs = ad.obs

        donor_n = None
        if "donor_id" in obs.columns:
            donor_n = obs["donor_id"].dropna().astype(str).nunique()

        age_min, age_max = None, None
        if "age" in obs.columns:
            age_min, age_max = extract_age_min_max(obs["age"])

        rows.append(
            {
                "dataset": name,
                "unique_donor_id_count": donor_n,
                "age_min": age_min,
                "age_max": age_max,
            }
        )

    result = pd.DataFrame(rows).set_index("dataset")
    print("\nDataset summary:")
    print(result)
    return result


def update_adata(ad, name):
    if "counts" not in ad.layers:
        raise KeyError(f"{name} is missing layers['counts']. Current layers: {list(ad.layers.keys())}")

    ad.X = ad.layers["counts"].copy()

    missing = [c for c in KEEP_COLS if c not in ad.obs.columns]
    for c in missing:
        ad.obs[c] = pd.NA
    ad.obs = ad.obs[KEEP_COLS].copy()

    return missing, ad.X.shape


def standardize_obs_and_x(datasets):
    report = []
    for name, ad in datasets.items():
        missing, shape = update_adata(ad, name)
        report.append((name, shape, missing))

    print("\nAfter setting X=counts and standardizing obs:")
    for name, shape, missing in report:
        miss_str = "none" if len(missing) == 0 else ",".join(missing)
        print(f"{name}: X shape={shape}; added missing obs columns: {miss_str}")


def strip_to_counts_and_obs(ad, name):
    if "counts" not in ad.layers:
        raise KeyError(f"{name} is missing layers['counts'], cannot keep counts only.")

    ad.X = ad.layers["counts"].copy()

    missing = [c for c in KEEP_COLS if c not in ad.obs.columns]
    for c in missing:
        ad.obs[c] = pd.NA
    ad.obs = ad.obs[KEEP_COLS].copy()

    ad.var = pd.DataFrame(index=ad.var_names.copy())
    ad.uns = {}

    for k in list(ad.obsm.keys()):
        del ad.obsm[k]
    for k in list(ad.obsp.keys()):
        del ad.obsp[k]
    if hasattr(ad, "varm"):
        for k in list(ad.varm.keys()):
            del ad.varm[k]
    if hasattr(ad, "varp"):
        for k in list(ad.varp.keys()):
            del ad.varp[k]

    for k in list(ad.layers.keys()):
        del ad.layers[k]

    try:
        ad.raw = None
    except Exception:
        pass

    return {
        "shape_X": tuple(ad.shape),
        "missing_obs_cols_filled": missing,
    }


def strip_all_datasets(datasets):
    report = {}
    for name, ad in datasets.items():
        report[name] = strip_to_counts_and_obs(ad, name)

    gc.collect()

    print("\nAfter stripping datasets to counts + selected obs:")
    for name, info in report.items():
        print(f"{name}: shape={info['shape_X']}, missing_obs_cols_filled={info['missing_obs_cols_filled']}")

    return report


def filter_adata10_by_age(datasets, low=19, high=97):
    adata10 = datasets["data10"]
    before = adata10.n_obs

    ages_num = pd.to_numeric(adata10.obs["age"].astype(str), errors="coerce")
    mask = (ages_num >= low) & (ages_num <= high)
    adata10 = adata10[mask].copy()
    adata10.obs["age"] = pd.to_numeric(adata10.obs["age"].astype(str), errors="coerce").astype("Int64")

    after = adata10.n_obs
    datasets["data10"] = adata10

    print(f"\ndata10 age filter: {before} -> {after} cells kept (age in [{low}, {high}])")


def convert_age_to_float(datasets):
    print("\nConverting age to float64:")
    for name, ad in datasets.items():
        if "age" not in ad.obs.columns:
            print(f"{name}: 'age' not found, skip")
            continue

        ad.obs["age"] = pd.to_numeric(ad.obs["age"].astype(str), errors="coerce").astype("float64")

        print(
            f"{name}: dtype={ad.obs['age'].dtype}; "
            f"range=({ad.obs['age'].min()}, {ad.obs['age'].max()}); "
            f"NaN count={ad.obs['age'].isna().sum()}"
        )


def align_and_concat_datasets(datasets):
    for k in ORDER_KEYS:
        ad = datasets[k]
        ad.var_names = ad.var_names.astype(str)
        if ad.var_names.has_duplicates:
            dup_n = ad.var_names.duplicated().sum()
            print(f"{k}: detected {dup_n} duplicated gene names, applying var_names_make_unique()")
            ad.var_names_make_unique()

    shared = None
    for k in ORDER_KEYS:
        genes = set(datasets[k].var_names)
        shared = genes if shared is None else (shared & genes)

    if not shared:
        raise ValueError("No overlapping genes found across datasets.")

    shared_order = [g for g in datasets["data1"].var_names if g in shared]
    shared_n = len(shared_order)
    print(f"\nTotal shared genes across datasets: {shared_n}")

    for k, label in zip(ORDER_KEYS, LABELS):
        ad = datasets[k][:, shared_order].copy()
        ad.obs["cohort"] = label
        datasets[k] = ad

    gc.collect()

    adata_all = sc.concat(
        [datasets[k] for k in ORDER_KEYS],
        join="inner",
        label=None,
        keys=LABELS,
        index_unique="-",
    )

    gc.collect()
    print(f"Concatenated adata_all shape: {adata_all.shape}")
    return adata_all


def inspect_gene_overlap(adata_all):
    genes_list = pd.read_csv(SHARED_GENES_FILE, header=None)[0]
    genes_list = genes_list.astype(str).str.strip()
    genes_set = pd.Index(genes_list.unique())

    adata_genes = pd.Index(adata_all.var_names.astype(str))

    overlap = adata_genes.intersection(genes_set)
    missing_from_adata = genes_set.difference(adata_genes)
    extra_in_adata = adata_genes.difference(genes_set)

    print("\nGene overlap summary:")
    print(f"Table gene count: {len(genes_set)}")
    print(f"adata_all gene count: {adata_all.n_vars}")
    print(f"Overlap count: {len(overlap)}")
    print(f"Missing from adata_all: {len(missing_from_adata)}")
    print(f"Extra in adata_all: {len(extra_in_adata)}")


def subset_to_shared_gene_file(adata_all):
    shared_genes = pd.read_csv(SHARED_GENES_FILE, header=None)[0].astype(str).values
    mask = adata_all.var_names.astype(str).isin(shared_genes)
    adata_all = adata_all[:, mask].copy()
    print(f"\nAfter subsetting to shared gene file: {adata_all.shape}")
    return adata_all


def finalize_and_save(adata_all):
    adata_all.obs["orig.ident"] = adata_all.obs["orig.ident"].astype(str)
    adata_all.write_h5ad(OUTPUT_FILE, compression="gzip")
    print(f"\nSaved merged AnnData to:\n{OUTPUT_FILE}")


def main():
    datasets = load_datasets()

    summarize_datasets(datasets)

    standardize_obs_and_x(datasets)

    strip_all_datasets(datasets)

    filter_adata10_by_age(datasets, low=19, high=97)

    convert_age_to_float(datasets)

    adata_all = align_and_concat_datasets(datasets)

    inspect_gene_overlap(adata_all)

    adata_all = subset_to_shared_gene_file(adata_all)

    finalize_and_save(adata_all)

    gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()
