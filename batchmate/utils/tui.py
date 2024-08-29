import time
import logging

from rich.progress import Progress

class TUI:
    def __init__(self, progress:Progress, epochs:int, train_size:int, eval_size:int) -> None:
        self.progress = progress
        self.epoch_bar = progress.add_task("[green]Epoch", total=epochs)
        self.train_bar = progress.add_task("[magenta]Training", total=train_size)
        self.eval_bar = progress.add_task("[cyan]Evaluating", total=eval_size)

    def reset_bars(self) -> None:
        self.progress.reset(self.train_bar)
        self.progress.reset(self.eval_bar)

    def step_epoch(self) -> None:
        self.progress.update(self.epoch_bar, advance=1)
        self.reset_bars()
    
    def step_train_batch(self) -> None:
        self.progress.update(self.train_bar, advance=1)
    
    def step_eval_batch(self) -> None:
        self.progress.update(self.eval_bar, advance=1)

if __name__ == '__main__':    
    import time
    import logging

    from rich.logging import RichHandler
    from rich.progress import (Progress,
                            SpinnerColumn,
                            BarColumn,
                            TaskProgressColumn,
                            TimeRemainingColumn)

    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    log = logging.getLogger("rich")

    epochs = 10
    train_size = 100
    eval_size = 50
    with  Progress(
            "{task.description}",
            SpinnerColumn(),
            BarColumn(bar_width=200),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
        tui = TUI(progress=progress, epochs=epochs, train_size=train_size, eval_size=eval_size)

        for epoch in range(1, epochs + 1):
            log.info(f"Epoch: {epoch}/{epochs}")

            for i in range(train_size):
                tui.step_train_batch()
                time.sleep(0.02)

            for i in range(eval_size):
                tui.step_eval_batch()
                time.sleep(0.02)

            tui.step_epoch()