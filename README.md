# BatchMate: Streamlining PyTorch Training Workflows

BatchMate is a Python library designed to simplify and enhance the process of training deep learning models using PyTorch. It provides a structured framework for managing training loops, logging, checkpointing, early stopping, and more.

## Features

- **Structured Training**: Simplifies the training loop with customizable hooks for loss calculation, accuracy evaluation, and more.
- **Logging**: Integrates with Rich for enhanced logging and supports WandB for experiment tracking.
- **Early Stopping**: Built-in support for early stopping based on validation loss.
- **Modular and Extensible**: Designed to be easily extended for custom use cases.

## Installation

### Prerequisites

Ensure you have Python 3.8 or later installed. [Poetry](https://python-poetry.org/) is used to manage dependencies and packaging.

### Install BatchMate

To install BatchMate, clone the repository and install the dependencies using Poetry:

```bash
git clone https://github.com/yourusername/batchmate.git
cd batchmate
poetry install
```

This will create a virtual environment and install all necessary dependencies.

## Usage

### Example: Training an MNIST Model

Here's an example of how to use BatchMate to train a simple convolutional neural network on the MNIST dataset.

1. **Navigate to the `examples` directory:**

    ```bash
    cd examples
    ```

2. **Run the `mnist.py` script:**

    ```bash
    poetry run python mnist.py --wandb --stop-loss
    ```

    This script will train the model using the MNIST dataset with WandB logging enabled and early stopping based on validation loss.

### Command-Line Arguments

- `--batch-size`: Input batch size for training (default: 64).
- `--epochs`: Number of epochs to train (default: 14).
- `--lr`: Learning rate (default: 1.0).
- `--gamma`: Learning rate step gamma (default: 0.7).
- `--no-cuda`: Disables CUDA training.
- `--seed`: Random seed (default: 1).
- `--wandb`: Enables logging to WandB.
- `--stop-loss`: Enables early stopping based on validation loss.
- `--stop-loss-patience`: Number of epochs with no improvement before stopping (default: 4).

### Saving the Model

The model will be automatically saved to the output directory specified in the `BatchMateConfig`. The default path is `./output/model.pth`.

### Customizing Your Trainer

BatchMate is designed to be extensible. You can easily customize the `MNISTrainer` class to implement your own training logic, loss functions, accuracy metrics, and more.

## Future Plans

- **Advanced Logging**: Add support for logging model graphs, gradient tracking, and other advanced metrics.
- **Custom Callbacks**: Implement user-defined callbacks for even more flexibility during training.
- **Dataset Augmentation**: Integrate advanced data augmentation pipelines.
- **Pre-trained Models**: Add support for loading pre-trained models and resuming from checkpoints.
- **Model Explainability**: Integrate tools for model explainability and interpretability.

## Contributing

We welcome contributions! If you have ideas, suggestions, or find any bugs, please feel free to open an issue or submit a pull request.

### TODOs

- [ ] Implement advanced logging features.
- [ ] Add support for dataset augmentation.
- [ ] Provide examples for more complex use cases.
- [ ] Improve documentation with more detailed examples and use cases.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for providing the deep learning framework.
- [Rich](https://github.com/Textualize/rich) for enhancing the terminal output.
- [WandB](https://wandb.ai/) for experiment tracking.

---