# A PyTorch Framework for Classification Tasks

This is a PyTorch-based framework for classification tasks.

## Project Structure

```
Classification_Framework_PyTorch
├── args.py
├── data.py
├── dataset
│   └── CIFAR100
├── main.py
├── ModelEvaluator.py
├── ModelTrainer.py
├── models
│   ├── __init__.py
│   └── ResNet.py
├── README.md
├── requirements.txt
├── saved_models
│   └── ResNet18-CIFAR100.bin
└── Visdom.py
```

The project consists of the following files and directories:

- `args.py`: Contains argument parsing logic using `argparse` for configuring model training.
- `data.py`: Defines custom dataset classes and functions for loading and preprocessing data.
- `dataset/`: Directory for storing dataset files, eg. `CIFAR100`.
- `main.py`: Main entry point for the project, orchestrating model training and testing.
- `ModelEvaluator.py`: Defines a model evaluator class for evaluating model performance, eg. accuracy.
- `ModelTrainer.py`: Defines a model trainer class for training the classification model.
- `models/`: Directory for storing model definitions.
  - `__init__.py`: The initialize file for the models package.
  - `ResNet.py`: Defines a template for the classification model, eg. `ResNet`.
- `saved_models/`: Directory for storing saved model weights. Format: `Model-Dataset.bin`, eg. `ResNet18-CIFAR100.bin`
- `Visdom.py`: Wrapper class for real-time data visualization using `visdom`.

## Usage

To train a model, run the `main.py` script with appropriate command-line arguments.

Example 1: Train model `ResNet50` on `CIFAR100` dataset:

```bash
python main.py --model ResNet50 --no_saving --no_plot # Default mode train, default dataset CIFAR100
```

If you need to draw curves, you must install and start  `visdom` application (view your plots locally at `http://localhost:8097`):

```bash
pip install visdom
visdom
```

Example 2: Test `ResNet18` and `ResNet34` models on `CIFAR100` dataset (ensure that you have trained models on `saved_models`):

```bash
python main.py --mode test --test_models ResNet18 ResNet34 # Space as separator
```
