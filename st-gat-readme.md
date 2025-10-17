# Spatio-Temporal Graph Attention Networks for Traffic Forecasting

This repository implements a Spatio-Temporal Graph Attention Network (ST-GAT) for short-term traffic forecasting, adapted from the framework proposed by:

> Bing Yu, Haoteng Yin, and Zhanxing Zhu. **Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting**. IJCAI 2018.

## Overview

The project predicts future traffic speeds using data from the PeMSD7 dataset. It models:

- **Spatial dependencies** between road sensors using graph attention mechanisms
- **Temporal dynamics** through historical time windows

## Repository Structure

```
├── data_loader/
│   ├── dataloader.py       # Creates spatio-temporal graph datasets
│   └── st_dat.py
├── models/
│   ├── st_gat.py           # ST-GAT model definition
│   └── trainer.py          # Training, evaluation, visualization
├── utils/
│   └── math_utils.py       # Normalization and metrics
├── dataset/
│   ├── PeMSD7_V_228.csv    # Traffic speed data
│   └── PeMSD7_W_228.csv    # Distance matrix
└── main.py                 # Entry point for training and evaluation
```

## Configuration

Model and data parameters are defined in `main.py`:

```python
config = {
    'BATCH_SIZE': 50,
    'EPOCHS': 50,
    'INITIAL_LR': 3e-4,
    'N_PRED': 9,
    'N_HIST': 12,
    'SPLITS': (34, 5, 5),
    'USE_GAT_WEIGHTS': True,
}
```

## Usage

### 1. Install Dependencies

```bash
pip install torch torch-geometric pandas numpy matplotlib tqdm
```

### 2. Prepare Dataset

Place `PeMSD7_V_228.csv` and `PeMSD7_W_228.csv` in the `dataset/` folder.

### 3. Run

```bash
python main.py
```

The script trains the ST-GAT model and generates a plot (`predictions_improved.png`) comparing predicted and actual traffic speeds.

## Evaluation

Performance is reported using:

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Square Error
- **MAPE** – Mean Absolute Percentage Error

Example output:

```
[Val] MAE: 3.21, RMSE: 5.81, MAPE: 7.2%
[Test] MAE: 3.15, RMSE: 5.63, MAPE: 6.9%
```

## Reference

Bing Yu, Haoteng Yin, and Zhanxing Zhu. **Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting**. Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI), 2018.
