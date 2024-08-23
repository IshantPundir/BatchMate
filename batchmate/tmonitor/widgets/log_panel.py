import os
import time
import threading
from logging import Logger
from logging.handlers import BufferingHandler

from rich.layout import Layout
from rich.panel import Panel

class LogPanel:
    def __init__(self, parent_layout:Layout, logger:Logger, capacity:int=100, update_interval:float=0.1) -> None:
        self._running = False
        self._parent_layout = parent_layout
        self._update_interval = update_interval

        self._buffering_handle = BufferingHandler(capacity=capacity)
        logger.addHandler(self._buffering_handle)
        
        self._panel = Panel("No logs yet", title="Log messages")

    def update_layout(self) -> None:
        """ This needs to be called to render table to the parent layout. """
        self._parent_layout.update(self._panel)

    def _wrap_text(self, text: str, width: int) -> str:
        wrapped_text = []
        current_line_length = 0
        start = 0
        
        for j, char in enumerate(text):
            if char == '\n':
                wrapped_text.append(text[start:j+1])
                start = j + 1
                current_line_length = 0
            elif current_line_length == width - 1:
                wrapped_text.append(text[start:j+1] + '\n')
                start = j + 1
                current_line_length = 0
            else:
                current_line_length += 1

        # Append the remaining part of the string
        if start < len(text):
            wrapped_text.append(text[start:])
        return wrapped_text
    
    def _update_log_panel(self) -> None:
        # TODO: Calculate panel width as well
        char_per_line = round(os.get_terminal_size().columns / 2)
        n_rows = os.get_terminal_size().lines - 10
        _log_messages = []

        for li in self._buffering_handle.buffer:
            log_message = li.msg
            log_message = "[bold green]$> [/]" + log_message
            # Make sure to wrap logs;
            if len(log_message) > char_per_line:
                log_message = self._wrap_text(text=log_message, width=char_per_line)
                _log_messages.extend(log_message)
            else:
                _log_messages.append(log_message)

        visible_logs = "\n".join(_log_messages[-n_rows:])        
        self._panel = Panel(visible_logs, title="Log messages")
        self.update_layout()

    def _update_loop(self) -> None:
        """Thread loop to continuously update the log panel."""
        while self._running:
            self._update_log_panel()
            time.sleep(self._update_interval)

    def run(self) -> None:
        """Start a thread that calls _update_log_panel every  update_interval."""
        self._running = True
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()

    def stop(self) -> None:
        """Stops the update loop."""
        self._running = False