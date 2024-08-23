import os

from rich.table import Table
from rich.layout import Layout
from rich.console import Console

class LogTable:
    """
    This is a dynamically updating rich table;
    you can add new row using `add_row()` method;
    It will dynamically update new labels and row entries
    """
    def __init__(self, parent_layout:Layout, *labels) -> None:
        self._labels = [*labels] # Labels to show; this will dynamically update, when new key is detected;
        self._row_data = []
        self._last_row_id = 0
        self._parent_layout = parent_layout

        self._console = Console()
        self._table = self._draw_table() # draw a basic table;

    def update_layout(self) -> None:
        """ This needs to be called to render table to the parent layout. """
        self._parent_layout.update(self._table)

    def _draw_table(self) -> Table:
        table = Table(title="Training progress",
                      expand=True,
                      show_lines=True)

        for label in self._labels:
            table.add_column(label, justify="left", style="cyan")            

        n_rows = round((os.get_terminal_size()[1] - 7) / 2)
        for row in self._row_data[-n_rows:]:
            table.add_row(*row)
        return table
    
    def _draw_and_update_table(self) -> None:
        """ Basically combines _draw_table and update_layout methods """
        self._table = self._draw_table()
        self.update_layout()

    # TODO: update data type from dict to LOG
    def add_data(self, row_id:int, data:dict) -> None:
        """
        
        Add row to table; 
        Arguments:
            row_id: (int) this is required so we can update an already rendered row with new data entry;
            data: (dict) dict with keys as labels and values as data entry;
        """
        # Check if any unknown labels are present in the data;
        for label in data:
            if label not in self._labels:
                self._labels.append(label)

        row_entry = []
        for i, label in enumerate(self._labels):
            if label in data:
                row_entry.append(str(data[label]))
            else:
                row_entry.append(self._row_data[-1][i] if row_id == self._last_row_id else '-')
        
        # Check if the row_id is same as last row id;
        # if true, delete the last row entry;
        if row_id == self._last_row_id:
            del self._row_data[-1]
        
        self._row_data.append(row_entry) # Append new row_entry to _row_data;
        self._draw_and_update_table() # Update table;
        self._last_row_id = row_id # update _last_row_id
