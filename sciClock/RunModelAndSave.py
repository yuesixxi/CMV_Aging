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

import os
import gc
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau




adata_train = sc.read_h5ad("/your/path/adata_train.h5ad")
adata_internal_test = sc.read_h5ad("/your/path/adata_internal_test.h5ad")
adata_external_test = sc.read_h5ad("/your/path/adata_external_test.h5ad")



class AgeDataset(Dataset):
    def __init__(self, adata, target_celltype="CD8T", scaler=None, is_train=True):
        adata = adata[adata.obs["Major_CT"] == target_celltype].copy()

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        y = adata.obs["age"].astype(np.float32).values
        donor_ids = adata.obs["donor_id"].astype(str).values

        if is_train:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            if scaler is None:
                raise ValueError("scaler must be provided when is_train=False")
            X = scaler.transform(X)
            self.scaler = scaler

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.donor_ids = donor_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.donor_ids[idx]


def build_dataloaders_with_val(
    adata_train,
    adata_val,
    adata_test,
    target_celltype="CD8T",
    batch_size=128,
    shuffle_train=True,
):
    train_dataset = AgeDataset(
        adata_train, target_celltype=target_celltype, is_train=True
    )
    scaler = train_dataset.scaler

    val_dataset = AgeDataset(
        adata_val, target_celltype=target_celltype, scaler=scaler, is_train=False
    )
    test_dataset = AgeDataset(
        adata_test, target_celltype=target_celltype, scaler=scaler, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        val_dataset.donor_ids,
        test_dataset.donor_ids,
    )


class AgePredictorVV3_Improved(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        self.block1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        self.block3_linear = nn.Linear(256, 256)
        self.block3_norm = nn.LayerNorm(256)
        self.block3_dropout = nn.Dropout(0.2)
        self.out_layer = nn.Linear(256, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        res = x
        x = self.block3_linear(x)
        x = self.block3_norm(x)
        x = self.block3_dropout(x)
        x = torch.nn.functional.silu(x)
        x = x + res
        latent = x
        return self.out_layer(latent), latent


def correlation_loss(y_true, y_pred):
    x = y_pred.squeeze()
    y = y_true.squeeze()

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.norm(vx) * torch.norm(vy) + 1e-8)
    return 1 - corr


def evaluate_model(model, loader, device):
    model.eval()
    preds, trues, donors = [], [], []

    with torch.no_grad():
        for xb, yb, donor_batch in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, _ = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
            donors.extend(donor_batch)

    y_pred = np.concatenate(preds).flatten()
    y_true = np.concatenate(trues).flatten()
    donor_ids = np.array(donors)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)

    return mse, mae, r, y_true, y_pred, donor_ids


def lin_ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    numerator = 2 * cov
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def donor_level_metrics(y_true, y_pred, donor_ids):
    donor_dict = {}
    for y_t, y_p, donor in zip(y_true, y_pred, donor_ids):
        donor_dict.setdefault(donor, {"true": [], "pred": []})
        donor_dict[donor]["true"].append(y_t)
        donor_dict[donor]["pred"].append(y_p)

    donor_trues, donor_preds = [], []
    for donor, values in donor_dict.items():
        if len(set(values["true"])) != 1:
            raise ValueError(f"Inconsistent age for donor {donor}")
        donor_trues.append(values["true"][0])
        donor_preds.append(np.mean(values["pred"]))

    donor_trues = np.array(donor_trues)
    donor_preds = np.array(donor_preds)

    donor_mse = mean_squared_error(donor_trues, donor_preds)
    donor_mae = mean_absolute_error(donor_trues, donor_preds)
    donor_r, _ = pearsonr(donor_trues, donor_preds)
    donor_ccc = lin_ccc(donor_trues, donor_preds)

    return donor_mse, donor_mae, donor_r, donor_ccc, donor_trues, donor_preds


def train_one_celltype(
    adata_train,
    adata_internal_test,
    adata_external_test,
    target_celltype,
    output_dir,
    batch_size=128,
    epochs=6,
    alpha=0.5,
    lr=2e-4,
    weight_decay=1e-5,
):
    print(f"\n========== Training for {target_celltype} ==========")

    train_loader, val_loader, test_loader, donor_ids_val, donor_ids_test = \
        build_dataloaders_with_val(
            adata_train,
            adata_internal_test,
            adata_external_test,
            target_celltype=target_celltype,
            batch_size=batch_size,
        )

    if len(train_loader.dataset) == 0:
        print(f"[Skip] No training cells found for {target_celltype}")
        return None, adata_external_test

    if len(val_loader.dataset) == 0:
        print(f"[Skip] No validation cells found for {target_celltype}")
        return None, adata_external_test

    if len(test_loader.dataset) == 0:
        print(f"[Skip] No test cells found for {target_celltype}")
        return None, adata_external_test

    input_dim = adata_train.X.shape[1]
    model = AgePredictorVV3_Improved(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-5,
    )

    torch.cuda.empty_cache()
    gc.collect()

    best_val_mse = float("inf")
    best_state_dict = None
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            noise = torch.empty_like(yb).uniform_(-1.5, 1.5)
            yb_noisy = yb + noise

            pred, latent = model(xb)

            loss_mse = criterion(pred, yb_noisy)
            loss_corr = correlation_loss(yb, pred)
            loss = loss_mse + alpha * loss_corr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)

        epoch_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item() * xb.size(0)

        epoch_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val MSE: {epoch_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if epoch_val_loss < best_val_mse:
            best_val_mse = epoch_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{target_celltype}.pt")
    torch.save(model.state_dict(), model_path)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train MSE+Corr+Noise")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train & Validation Loss - {target_celltype}")
    plt.legend()
    plt.tight_layout()

    loss_plot_path = os.path.join(output_dir, f"loss_curve_{target_celltype}.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    mse, mae, r, y_true, y_pred, donor_ids = evaluate_model(model, test_loader, device)
    donor_mse, donor_mae, donor_r, donor_ccc, donor_trues, donor_preds = donor_level_metrics(
        y_true, y_pred, donor_ids
    )

    mask = adata_external_test.obs["Major_CT"] == target_celltype
    adata_external_test.obs.loc[mask, f"PreAge_{target_celltype}"] = y_pred
    adata_external_test.obs.loc[mask, f"AgeAcceleration_{target_celltype}"] = (
        adata_external_test.obs.loc[mask, f"PreAge_{target_celltype}"]
        - adata_external_test.obs.loc[mask, "age"]
    )

    metrics = {
        "celltype": target_celltype,
        "cell_level_mse": mse,
        "cell_level_mae": mae,
        "cell_level_pearson_r": r,
        "donor_level_mse": donor_mse,
        "donor_level_mae": donor_mae,
        "donor_level_pearson_r": donor_r,
        "donor_level_ccc": donor_ccc,
        "n_train_cells": len(train_loader.dataset),
        "n_val_cells": len(val_loader.dataset),
        "n_test_cells": len(test_loader.dataset),
    }

    print(f"[Done] {target_celltype}")
    print(metrics)

    return metrics, adata_external_test


def main():
    celltypes = ["NK", "CD8T", "CD4T", "MONO", "B"]
    output_dir = "/vol/projects/age_model_outputs"
    batch_size = 128
    epochs = 6
    alpha = 0.5

    results = []

    for target_celltype in celltypes:
        metrics, adata_external_test_updated = train_one_celltype(
            adata_train=adata_train,
            adata_internal_test=adata_internal_test,
            adata_external_test=adata_external_test,
            target_celltype=target_celltype,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
        )

        if metrics is not None:
            results.append(metrics)

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "metrics_summary.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Saved metrics summary to: {results_csv}")

    h5ad_path = os.path.join(output_dir, "externaltest_with_predictions.h5ad")
    adata_external_test.write(h5ad_path, compression="gzip")
    print(f"Saved updated AnnData to: {h5ad_path}")


if __name__ == "__main__":
    main()
