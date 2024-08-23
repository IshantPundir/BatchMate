from rich.layout import Layout
from rich.progress import (Progress,
                           SpinnerColumn,
                           BarColumn,
                           TaskProgressColumn,
                           TimeRemainingColumn)


class ProgressWindow:
    def __init__(self, parent_layout:Layout, epochs:int) -> None:
        self.parent_layout = parent_layout

        self.job_progress = Progress(
            "{task.description}",
            SpinnerColumn(),
            BarColumn(bar_width=200),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True
        )

        self.job_progress.add_task("[green]Epoch", total=epochs) # Task id: 0
        self.job_progress.add_task("[magenta]Training", total=100) # Task id: 1
        self.job_progress.add_task("[cyan]Testing", total=100) # Task id: 2
    
    def _step_bar(self, id:int) -> None:
        progress_bar = self.job_progress.tasks[id]
        if not progress_bar.finished:
            self.job_progress.advance(progress_bar.id)
            
    def update_layout(self) -> None:
        """ This needs to be called to render table to the parent layout. """
        self.parent_layout.update(self.job_progress)

    def set_total_epochs(self, epochs:int) -> None:
        self.job_progress.reset(task_id=0, total=epochs)

    def set_train_batch(self, n_batch:int) -> None:
        self.job_progress.reset(task_id=1, total=n_batch)
        self.job_progress.refresh()
    
    def set_test_batch(self, n_batch: int) -> None:
        self.job_progress.reset(task_id=2, total=n_batch)
        self.job_progress.refresh()

    def step_epochs(self) -> None:
        self._step_bar(0)
    
    def step_train_batch(self) -> None:
        self._step_bar(1)
    
    def step_test_batch(self) -> None:
        self._step_bar(2)
