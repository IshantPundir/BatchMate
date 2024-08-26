import os
import time
import logging
import threading
from collections import deque

from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.logging import RichHandler


class LogReader():
    _instance = None
    def __init__(self):
        self.messages = deque([])
        self.size = 100

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def write(self, message):
        self.messages.extend(message.splitlines())
        while len(self.messages) > self.size:
            self.messages.popleft()

    def flush(self):
        pass


class LogPanel:
    def __init__(self, parent_layout:Layout, update_interval:float=0.2) -> None:
        self._parent_layout = parent_layout
        self.update_interval = update_interval
        self._running = False
        self._log_std = LogReader()
        
        c = Console(file=LogReader())
        r = RichHandler(console=c, rich_tracebacks=True, tracebacks_show_locals=True,

        FORMAT = "%(message)s"
        # FORMAT = "%(asctime)-15s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level="NOTSET",
            format=FORMAT,
            datefmt="[%X]",
            handlers=[r],
        )

        self._panel = Panel("No logs yet!", title="Logs")
        


    def update_layout(self) -> None:
        self._parent_layout.update(self._panel)

    def _update(self) -> None:
        while self._running:
            n_rows = round((os.get_terminal_size()[1] - 7) / 2)
            self._log_std.size = n_rows

            text = self._log_std.messages
            text = "\n".join(text)

            self._panel = Panel(Text(text=text, overflow="fold", no_wrap=False))
            self.update_layout()
            time.sleep(self.update_interval)

    def run(self) -> None:
        """Start a thread that calls _update_log_panel every  update_interval."""
        self._running = True
        update_thread = threading.Thread(target=self._update, daemon=True)
        update_thread.start()

    def stop(self) -> None:
        """Stops the update loop."""
        self._running = False