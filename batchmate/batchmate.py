import os
import inspect
import logging
from typing import Tuple, Optional
from abc import ABC, abstractmethod

import wandb
import torch
from torch import nn
from rich.live import Live
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from batchmate.utils import Log, StopLoss
from batchmate.tmonitor import TMonitor

class BatchMate(ABC):
    """
    BatchMate is an abstract class that simplifies the training, testing, and monitoring
    of DNN models. 
    """
    def __init__(self, run_name:str, model:nn.Module,
                 train_dataloader:DataLoader, test_dataloader:DataLoader,
                 optimizer:Optimizer, scheduler:Optional[Optimizer]=None,
                 stop_loss:bool=False, stop_loss_patience:int=7, stop_loss_verbose:bool=False, stop_loss_delta:float=0.0,
                 checkpoint_duration:int=10,
                 output_dir:str|None=None, device:str="cuda", show_tui:bool=True,
                 log_to_wandb:bool=False, wandb_project="VisionCore", acc_per_batch:bool=True) -> None:
        self.device = device
        self.model = model.to(self.device) # Store the model and move it the device;
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.show_tui = show_tui
        self.stop_loss = stop_loss
        self.acc_per_batch = acc_per_batch
        self.checkpoint_duration = checkpoint_duration

        # Initialize TUI
        if self.show_tui is True:
            self.tui = TMonitor()

        # Create output directory with run_name;
        self.output_dir, self.ckpt_dir = self._init_output_dir(output_dir=output_dir,
                                                               run_name=run_name)
        
        # Initialize StopLoss if required
        if self.stop_loss is True:
            # TODO: pass logger as trac_func arg. 
            self.check_stop_loss = StopLoss(patience=stop_loss_patience,
                                            delta=stop_loss_delta,
                                            verbose=stop_loss_verbose)
        # Initialize wandb
        if log_to_wandb:
            # TODO: track hyperparameters using config arg
            self.wandb_run = wandb.init(project=wandb_project, name=run_name)
            # wandb.watch(model)
        else: self.wandb_run = None

    def _init_output_dir(self, output_dir:str|None, run_name:str) -> Tuple[str, str]:
        """
        Create a directory tree like this: `output_dir/run_name/checkpoints`

        This returs the path the path to output_dir and ckpt_dir;
        """
        if output_dir is None:
            # Fetch the path to the file initializing BatchMate;
            stack = inspect.stack()
            caller_frame = stack[-1]
            caller_file = caller_frame.filename
            output_dir = os.path.dirname(caller_file)
        
        output_dir = os.path.join(output_dir, "output/", run_name)
        ckpt_dir = os.path.join(output_dir, 'checkpoints')        
        os.makedirs(ckpt_dir, exist_ok=True) # Create the run directories;

        return output_dir, ckpt_dir
    
    def log_images(self, image:torch.Tensor, key:str="images", caption:str="") -> None:
        """ Method to log images to wandb. """
        if self.wandb_run is None: return
        image = wandb.Image(image, caption=caption)
        self.wandb_run.log({key: image})

    def log_metric(self, epoch:int, log_type:str, metric:Log) -> None:
        ''' Simple method to log "Log" to wandb and console; '''
        metric.log_type = log_type
        logs = metric.log_with_type(ignore_key='epoch')

        # Log metric to wandb;
        if self.wandb_run is not None:
            self.wandb_run.log(logs)

        if self.show_tui is True:
            self.tui.training_logs.add_data(row_id=epoch, data=logs)
        else:
            print(f'LOG: {logs}')

    def training_steps(self, loss:torch.Tensor) -> None:
        """
        You may override this method if you need additional steps for training the models
        such as gradient clippping, etc...
        """
        # Back-propogation while training;
        loss.backward()
        # Step optimizer
        self.optimizer.step()

    
    def _batch_run(self, epoch:int, dataloader:DataLoader, training:bool=True) -> Tuple[Log, Log]:
        run_logs = Log(epoch=epoch)
        run_results = Log(epoch=epoch)
        
        # Update total batches for TUI progress bar!
        if self.show_tui is True:
            if training is True:
                self.tui.progress_view.set_train_batch(len(dataloader))
            else:
                self.tui.progress_view.set_test_batch(len(dataloader))

        for batch in dataloader:
            # Get the model's output along with true labels
            # or anything that's necessary to calculate loss and accuracy;
            if training is True:
                self.optimizer.zero_grad()

            _batch_results = self.batch_inference(batch, training=training)

            # Calculate Loss
            loss = self.loss_fn(_batch_results)
            
            if training is True:            
                self.training_steps(loss=loss)

            # Calculate accuracy if self.acc_per_batch is True
            if self.acc_per_batch is True:
                batch_accuracy = self.acc_fn(_batch_results)
                run_logs.append_log(batch_accuracy)

            # Append the loss value to training_logs
            run_logs.append_log(Log(loss=loss.detach().item()))
            
            # Append _batch_results to training_results
            # This will later be used to calculate accuracy;
            run_results.append_log(_batch_results)

            # Step the progress bar
            if self.show_tui:
                if training is True:
                    self.tui.progress_view.step_train_batch()
                else:
                    self.tui.progress_view.step_test_batch()

        return run_logs, run_results
    
    def _process_batch_run(self, logs:Log, results:Log) -> Log:
        logs_avg = logs.average_all_values()

        if self.acc_per_batch is False:
            # Calculate accuracy from entire train_results;
            acc = self.acc_fn(results)
            logs_avg.append_log(acc)

        return logs_avg
    
    @abstractmethod
    def batch_inference(self, batch, training:bool) -> Log:
        """
        This method is called for each batch of trainging and validation datasets;

        You need to degine how to extract input data and true labels from batch,
        and how this data is passed to the model;

        After inferencing you can return the model's output and expected output wrapped in Log class;

        Later this log will be used to calculate loss & accuracy;

        This is what a simple impliementaion of this method might look like;
        ```
        input_data, output_labels = batch['input'], batch['output']
        result = self.model(input_data)
        return Log(output_labels=output_labels, model_output=result)
        ```

        NOTE: We don't need to calculate loss/accuracy or perform any back-propagation here;
        """
        return Log()
    
    @abstractmethod
    def loss_fn(self, results:Log) -> torch.Tensor:
        """
        This methods is called immediately after batch_inference;
        model's output and labels to calculate loss are expected to be wrapped in Log;
        """
        return
    
    @abstractmethod
    def acc_fn(self, results:Log) -> Log:
        return
    
    @abstractmethod
    def save(self, path:str) -> Log:
        return
    
    def epoch_start_callback(self, epoch:int) -> None:
        """Optional method that can be overridden by subclasses."""
        pass

    def epoch_end_callback(self, train_logs:Log, train_results:Log, train_logs_avg:Log,
                           eval_logs:Log, eval_results:Log, eval_logs_avg:Log) -> None:
        """Optional method that can be overridden by subclasses."""
        pass
    
    def _run(self, epochs:int) -> None:
        for epoch in range(1, epochs + 1):
            print(f'Epoch: {epoch}')
            self.epoch_start_callback(epoch=epoch)
    
            # Training...
            self.model.train()
            train_logs, train_results = self._batch_run(epoch=epoch, dataloader=self.train_dataloader, training=True)
            train_logs_avg = self._process_batch_run(logs=train_logs, results=train_results)
            self.log_metric(epoch=epoch, log_type="train", metric=train_logs_avg)

            # Validating...
            self.model.eval()
            with torch.no_grad():
                eval_logs, eval_results = self._batch_run(epoch=epoch, dataloader=self.test_dataloader, training=False)
                eval_logs_avg = self._process_batch_run(logs=eval_logs, results=eval_results)
            self.log_metric(epoch=epoch, log_type="eval", metric=eval_logs_avg)

            # Check if StopLoss is enabled and triggered;
            if self.stop_loss is True:
                stop_training = self.check_stop_loss(model=self.model,
                                    optimizer=self.optimizer,
                                    val_loss=eval_logs_avg.loss)
                if stop_training is True:
                    # TODO: Stop the training...
                    ...

            # Call the epoch_end_callback with training and evaluation batrch run logs.
            self.epoch_end_callback(train_logs=train_logs,
                                    train_results=train_results,
                                    train_logs_avg=train_logs_avg,
                                    eval_logs=eval_logs,
                                    eval_results=eval_results,
                                    eval_logs_avg=eval_logs_avg)
            
            # Step the scheduler if provided;
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Saving model's checkpoints
            if epoch % self.checkpoint_duration == 0:
                checkpoint_name = f"{epoch + 1}_loss:{train_logs_avg.loss}.pth"
                save_path = os.path.join(self.ckpt_dir, checkpoint_name)
                self.save(save_path)

            if self.show_tui is True:
                self.tui.progress_view.step_epochs()
        
    def run(self, epochs:int) -> None:
        # Set the total numbe of epochs for TUI
        if self.show_tui is True:
            self.tui.progress_view.set_total_epochs(epochs)

            with Live(self.tui.layout, refresh_per_second=4, screen=True, vertical_overflow="visible") as tui_live:
                self._run(epochs=epochs)

            self.tui.close()

        else:
            self._run(epochs=epochs)