# CAAFP - Clustered Federated Learning for Human Activity Recognition


## Overview

CAAFP (Cluster-Aware Adaptive Federated Pruning) implements and compares multiple federated learning strategies designed for human activity recognition tasks. The project includes implementations of several approaches including cluster-based federated learning, efficient federated learning, and specialized pruning techniques.

## Code Content

- **Multiple FL Algorithms**: Implementations of various federated learning approaches
  - CAAFP (Cluster-Aware Adaptive Federated Pruning)-ours
  - ClusterFL (Clustering-based Federated Learning)
  - Efficient FL
  - FedChar
  - FedSNIP

- **Dataset Support**:
  - WISDM (Wireless Sensor Data Mining) dataset
  - UCI Human Activity Recognition dataset

- **CNN-based Models**: for activity recognition

## Repository Structure

```
├── main_caafp_cnn_cluster.py       # CAAFP implementation for WISDM
├── main_caafp_cnn_uci_cluster.py   # CAAFP implementation for UCI dataset
├── main_clusterfl_cnn.py           # ClusterFL implementation for WISDM
├── main_clusterfl_cnn_uci.py       # ClusterFL implementation for UCI dataset
├── main_efficient_fl_cnn.py        # Efficient FL for WISDM
├── main_efficient_fl_cnn_uci.py    # Efficient FL for UCI dataset
├── main_fedchar_cnn.py             # FedChar implementation for WISDM
├── main_fedchar_cnn_uci.py         # FedChar implementation for UCI dataset
├── main_fedsnip_cnn.py             # FedSNIP implementation for WISDM
├── main_fedsnip_cnn_uci.py         # FedSNIP implementation for UCI dataset
├── data_loader.py                  # Data loading utilities for WISDM
├── data_loader_uci.py              # Data loading utilities for UCI dataset
├── models_cnn.py                   # CNN model definitions for WISDM
├── models_cnn_uci.py               # CNN model definitions for UCI dataset
├── metrics.py                      # Communication cost tracker
├── run_wisdm.py # Benchmark script for WISDM
└── run_uci.py # Benchmark script for UCI dataset
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- scikit-learn
- pandas

### Setup

```bash
# Clone the repository
git clone https://github.com/OmGovind15/CAAFP.git
cd CAAFP

# Install required packages
pip install torch numpy scikit-learn pandas
```

## Usage

### Running CAAFP

For WISDM dataset:
```bash
python main_caafp_cnn_cluster.py
```

For UCI HAR dataset:
```bash
python main_caafp_cnn_uci_cluster.py
```

### Running Other Algorithms

Each implementation can be run similarly:

```bash
# ClusterFL
python main_clusterfl_cnn.py

# Efficient FL
python main_efficient_fl_cnn.py

# FedChar
python main_fedchar_cnn.py

# FedSNIP
python main_fedsnip_cnn.py
```

### Running Benchmarks

To compare multiple approaches:

```bash
# For WISDM dataset
python run_wisdm.py

# For UCI dataset
python run_uci.py
```
