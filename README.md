# Iris Classification with a Custom MLP

This project implements a **from-scratch Multi-Layer Perceptron (MLP)** for the Iris classification task and runs an **exhaustive grid search over 2,592 configurations**. The pipeline automatically exports results to **CSV**, generates a **LaTeX Top-40 table**, produces **16 bar plots** (hyperparameters vs. accuracy / best epoch), and prints a **distribution summary for all perfect configurations**.

## What’s Included

- **End-to-end experiment pipeline**: train → evaluate → save `results.json`
- **Analysis pipeline**: `JSON → CSV → LaTeX table → plots → console summary`
- **Metrics**: accuracy, precision, recall, F1-score + confusion matrix per experiment
- **SOLID/OOP analysis components**: converters, table generators, plot generators, analyzers (DI-friendly)

## Hyperparameter Grid (2,592 runs)

The grid search explores:

- **Data preparation**
  - `split_strategy`: `0.8/0.1/0.1`, `0.7/0.2/0.1`, `0.75/0.15/0.1`
  - `scaler_type`: `standard`, `minmax`
- **Model**
  - `layers`: `[32,16]`, `[64,32]`, `[128,64]`, `[64,32,16]` (and other defined variants)
  - `activation`: `relu`, `tanh`
- **Training**
  - `learning_rate`: `0.001`, `0.01`, `0.05`
  - `momentum`: `0.9`, `0.95`
  - `batch_size`: `8`, `16`, `32`
  - `max_epochs`: `30`, `50`, `100`

## Key Results (Summary)

Across all **2,592** configurations, performance varies strongly with hyperparameters. The project found **94 configurations with 100% test accuracy**.

**Distribution among perfect configurations (n=94):**
- **Split strategy**: `0.7/0.2/0.1` (54.3%), `0.8/0.1/0.1` (33.0%), `0.75/0.15/0.1` (12.8%)
- **Scaler**: `minmax` (97.9%), `standard` (2.1%)
- **Layers**: `[64,32,16]` (46.8%), `[32,16]` (34.0%), `[64,32]` (9.6%), `[128,64]` (9.6%)
- **Activation**: `tanh` (54.3%), `relu` (45.7%)
- **Learning rate**: `0.05` (71.3%), `0.01` (25.5%), `0.001` (3.2%)
- **Momentum**: `0.95` (57.4%), `0.9` (42.6%)
- **Batch size**: `8` (54.3%), `16` (33.0%), `32` (12.8%)

The outputs also include bar plots showing average **Test Accuracy** and **Best Epoch** per hyperparameter choice, plus a LaTeX table listing the Top-40 runs.