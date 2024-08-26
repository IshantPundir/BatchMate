"""
This is the TUI framework to display logs and progress while training models.

Do I need this??
no!

Do I want this??
YES!!!

Why?
It makes the training and testing process a little fun;
Instead of staring at boring terminal, it's a middle ground 
between advance wandb WebUI and plain terminal.
"""
import time
import logging

from rich.live import Live
from rich.layout import Layout

from batchmate.tmonitor.widgets import (LogTable,
                     LogPanel,
                     ProgressWindow)

class TMonitor:
    def __init__(self, epochs:int=100) -> None:
        # self.console = Console()
        self.layout = self._make_layout()

        self.training_logs = LogTable(parent_layout=self.layout['log-view']['training-log'])
        self.log_pannel = LogPanel(parent_layout=self.layout['log-view']['general-log-panel'])
        self.progress_view = ProgressWindow(parent_layout=self.layout['progress-view'], epochs=epochs)

        self.training_logs.update_layout()
        self.progress_view.update_layout()
        self.log_pannel.update_layout()
        
        # Start the log pannel thread;
        self.log_pannel.run()

    def _make_layout(self) -> Layout:
        layout = Layout() # root layout;
        
        layout.split_column(
            Layout(name="log-view", ratio=1),
            Layout(name="progress-view", size=3)
        )

        layout["log-view"].split_row(
            Layout(name="training-log"),
            Layout(name="general-log-panel", ratio=1)
        )

        return layout

    def close(self) -> None:
        self.log_pannel.stop()

if __name__ == '__main__':
    log = logging.getLogger(__name__)
    monitor = TMonitor()
    
    # Local logger
    try:
        epochs = 100
        train_batch_size = 10
        test_batch_size = 5
        monitor.progress_view.set_total_epochs(epochs)

        with Live(monitor.layout, refresh_per_second=4, screen=True, vertical_overflow="ellipsis") as live:            
            for epoch in range(1, epochs):
                # r.tracebacks_width = 80
                # Training loop
                monitor.progress_view.set_train_batch(train_batch_size)
                for _ in range(train_batch_size):
                    monitor.progress_view.step_train_batch()
                    time.sleep(0.1)

                monitor.training_logs.add_data(row_id=epoch, data={"epoch": epoch, "train loss": 0.14, "train accuracy": 0.56})

                # Testing loop:
                monitor.progress_view.set_test_batch(test_batch_size)
                for _ in range(test_batch_size):
                    monitor.progress_view.step_test_batch()
                    # log.info(f"testing...")
                    time.sleep(0.1)

                monitor.training_logs.add_data(row_id=epoch, data={"epoch": epoch, "test loss": 0.04, "test accuracy": 0.24})                            
                
                # Step epoch and log some data;
                monitor.progress_view.step_epochs()
                
                # try:
                #     print(1 / 0)
                # except Exception:
                #     log.exception("unable print!")
                
                log.info(f"Epoch done: {epoch} console_len: {live.console.size}")

        # monitor.run(epochs=100)
    except:
        monitor.close()

    monitor.close()
    