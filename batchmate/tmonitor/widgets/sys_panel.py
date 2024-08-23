import time
import psutil
import logging
import subprocess

from rich.layout import Layout
from rich.panel import Panel

class SystemView:
    def __init__(self, parent_layout:Layout, update_interval: float = 1.0) -> None:
        """
        Initialize the SystemView class.

        :param update_interval: Time in seconds between updates of the system stats.
        """
        self._parent_layout = parent_layout
        self.update_interval = update_interval
        self._running = False
        self._update_thread = None
        
        mem = psutil.virtual_memory()
        self._total_mem = self._bytes_to_gb(mem.total)

        self._panel = Panel("", title="System monitor")
 
    def update_layout(self) -> None:
        """ This needs to be called to render table to the parent layout. """
        self._parent_layout.update(self._panel)
 
    def _get_cpu_usage(self) -> str:
        """Returns the current CPU usage as a percentage."""
        return {
            "CPU Usage:": f"{psutil.cpu_percent(interval=None)}%"
        }

    def _get_ram_usage(self) -> str:
        """Returns the current RAM usage."""
        mem = psutil.virtual_memory()
        return {
            "Ram Usage:": f"{mem.percent}%",
            "Usage:": f"{self._bytes_to_gb(mem.used):.2f}/{self._total_mem:.2f}GB",
        }

    def _get_gpu_usage(self) -> str:
        """Returns the current GPU usage using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            gpu_info = result.stdout.strip().split(',')
            gpu_usage = gpu_info[0].strip()
            memory_usage = float(gpu_info[1].strip())
            gpu_space = float(gpu_info[2].strip())

            return {
                "GPU Usage": f"{gpu_usage}%",
                "Memory Usage:": f"{memory_usage / 1024:.2f}/{gpu_space / 1024:.2f}GB"
            } 
        
        except subprocess.CalledProcessError as e:
            return {"Error:": "Error fetching GPU stats"}
        except FileNotFoundError:
            return {"Error:": "nvidia-smi not found."}

    def _bytes_to_gb(self, bytes_value: int) -> float:
        """Converts bytes to gigabytes."""
        return bytes_value / (1024 ** 3)

    def _display_system_usage(self) -> None:
        """Prints the system usage for CPU, RAM, and GPU."""
        while self._running:
            print("\n" + "-"*40)
            print(self._get_cpu_usage())
            print(self._get_ram_usage())
            print(self._get_gpu_usage())
            print("-"*40 + "\n")
            time.sleep(self.update_interval)

    def run(self) -> None:
        """Start monitoring and displaying the system usage."""
        self._running = True
        # self._display_system_usage()
        # self._update_thread = threading.Thread(target=self._display_system_usage, daemon=True)
        # self._update_thread.start()

    def stop(self) -> None:
        """Stop monitoring the system usage."""
        self._running = False
