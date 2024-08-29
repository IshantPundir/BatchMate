"""
This is an Example of using Batchmate for training a simple MNIST model.
"""
import logging
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from batchmate import BatchMate
from batchmate.batchmate import BatchMateConfig, WandBConfig
from batchmate.utils import Log

log = logging.getLogger("rich")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class MNISTrainer(BatchMate):
    def __init__(self, run_name: str,
                 model: nn.Module,
                 device: str,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 optimizer: optim.Optimizer,
                 **kwargs) -> None:
        super().__init__(run_name, model, device, train_dataloader, test_dataloader, optimizer, **kwargs)
    
    def save(self, path: str) -> Log:
        return super().save(path)
    
    def acc_fn(self, results: Log) -> Log:
        model_output = results.model_output
        true_labels = results.true_labels
        pred = model_output.argmax(dim=1, keepdim=True)

        correct =  pred.eq(true_labels.view_as(pred)).sum().item()
        correct = 100. * correct / model_output.shape[0]
        return Log(acc=correct)
    
    def loss_fn(self, results: Log) -> torch.Tensor:
        model_output = results.model_output
        true_labels = results.true_labels
        loss = F.nll_loss(model_output, true_labels)
        return loss
    
    def epoch_end_callback(self, train_logs: Log, train_results: Log, train_logs_avg: Log,
                           eval_logs: Log, eval_results: Log, eval_logs_avg: Log) -> None:
        # Let's log input images and labels to wandb as well
        input_images = eval_results.model_input
        true_labels = eval_results.true_labels
        model_output = eval_results.model_output

        # Take first 5 images and make it a grid along with predicted and true labels;
        input_images = make_grid(input_images[-1][:5])
        model_output = model_output[-1][:5].argmax(dim=1, keepdim=False).to('cpu')
        true_labels = true_labels[-1][:5]

        # Log the images to wandb
        caption = f"True labels: {true_labels.to('cpu')} | Model output: {model_output.to('cpu')}"        
        self.log_images(image=input_images, caption=caption)

    def batch_inference(self, batch, training: bool) -> Log:
        images, labels = batch
        # Move values to device.
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Perform inference
        results = self.model(images)

        # Return the model's output, expected output and any other data
        # That we might need to calculate loss, accuracy or log later!
        return Log(model_input=images, model_output=results, true_labels=labels)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch BatchMate MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='log to WandB')
    parser.add_argument('--show-tui', action='store_true', default=False,
                        help='Show TUI')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Define the transformer
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the datasets
    train_dataset = datasets.MNIST('./examples/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./examples/data', train=False, transform=transform)

    # Wrap datasets aroud DataLoader
    dataloader_config = {'batch_size': args.batch_size}
    if use_cuda:
        dataloader_config.update({
            "num_workers": 1,
            "pin_memory": True,
        })

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_config)
    test_dataloader = DataLoader(test_dataset,  shuffle=False, **dataloader_config)


    # Initalize model, optimizer, scheduler, etc...
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Initialize trainer
    batchmate_config = BatchMateConfig()
        
    if args.wandb is True:
        batchmate_config.wandb_config = WandBConfig(project="MNIST") 

    trainer = MNISTrainer(run_name="MNIST", model=model, device="cuda",
                          train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                          optimizer=optimizer, scheduler=scheduler, config=batchmate_config)
    
    log.info("Let's start training boiiii!")
    
    # Let's start the training...
    trainer.run(epochs=args.epochs)