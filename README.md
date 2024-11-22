# Introduction to Deep Learning

Welcome to the "Introduction to Deep Learning" course repository. This repository contains materials, code, and resources to support your learning journey in deep learning using PyTorch, Lightning, and other essential libraries. It is still in the making, and will become more furnished as time goes.
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [License](#license)

## Introduction

This course covers the fundamentals of deep learning, including neural networks, training techniques, and practical applications. We utilize the following technologies:

- [PyTorch](https://pytorch.org/): An open-source machine learning library.
- [Lightning](https://lightning.ai/): A lightweight PyTorch wrapper for high-performance AI research.
- [Pandas](https://pandas.pydata.org/): A data manipulation and analysis library.
- [Matplotlib](https://matplotlib.org/): A plotting library for creating static, animated, and interactive visualizations.
- [Seaborn](https://seaborn.pydata.org/): A statistical data visualization library based on Matplotlib.
- [TensorBoard](https://www.tensorflow.org/tensorboard): A tool for providing the measurements and visualizations needed during the machine learning workflow.

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Malchemis/Introduction-to-Deep-Learning.git
   cd Introduction-to-Deep-Learning
   ```

2. **Run the setup script:**

   - Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.

   - Run the script `setup.sh` line by line.

   This script will create a conda environment named `iDL` and install all required packages.

## Project Structure

The repository is organized as follows:

```plaintext
Introduction-to-Deep-Learning/
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed datasets
├── notebooks/              # Jupyter notebooks
├── models/                 # Trained and serialized models
├── src/
│   ├── data/               # Data loading and processing scripts
│   ├── models/             # Model architectures
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation scripts
├── tests/                  # Unit tests
├── setup.sh                # Setup script for environment and dependencies
├── README.md               # Project overview and instructions
└── requirements.txt        # List of required packages
```

## Usage

After setting up the environment, you can start exploring the notebooks or run training scripts. For example, to run a training script:

```bash
python src/training/train_model.py
```

Ensure you activate the conda environment before running any scripts:

```bash
conda activate iDL
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.