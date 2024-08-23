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
                     SystemView,
                     ProgressWindow)

class TMonitor:
    def __init__(self, logger:logging.Logger, epochs:int=100) -> None:
        # self.console = Console()
        self.layout = self._make_layout()

        self.training_logs = LogTable(parent_layout=self.layout['log-view']['training-log'])
        self.progress_view = ProgressWindow(parent_layout=self.layout['progress-view'], epochs=epochs)
        self.log_pannel = LogPanel(parent_layout=self.layout['log-view']['general-view']['general-log-panel'], logger=logger)
        self.sys_pannel = SystemView(parent_layout=self.layout['log-view']['general-view']['system-monitor'])

        self.training_logs.update_layout()
        self.progress_view.update_layout()
        self.log_pannel.update_layout()
        self.sys_pannel.update_layout()
        
        # Start the log pannel thread;
        self.sys_pannel.run()
        self.log_pannel.run()

    def _make_layout(self) -> Layout:
        layout = Layout() # root layout;
        
        layout.split_column(
            Layout(name="log-view", ratio=1),
            Layout(name="progress-view", size=3)
        )

        layout["log-view"].split_row(
            Layout(name="training-log"),
            Layout(name="general-view")
        )

        layout["log-view"]["general-view"].split_column(
                Layout(name="general-log-panel", ratio=1),
                Layout(name="system-monitor", size=5)
        )

        return layout

    def close(self) -> None:
        self.log_pannel.stop()
        self.sys_pannel.stop()

if __name__ == '__main__':
    # Set up the main/root logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)
    log = logging.getLogger("rich")
    
    # Local logger
    monitor = TMonitor(logger=main_logger)
    try:
        epochs = 100
        train_batch_size = 10
        test_batch_size = 5
        monitor.progress_view.set_total_epochs(epochs)

        with Live(monitor.layout, refresh_per_second=4, screen=True, vertical_overflow="visible") as live:
            for epoch in range(1, epochs):
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
                log.info(f"Epoch [bold cyan]{epoch}[/] [green]Done![/]")

        # monitor.run(epochs=100)
    except:
        monitor.close()

    monitor.close()
    