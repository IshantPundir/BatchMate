from __future__ import annotations

class Log:
    """ Store results here to log to console and wandb """
    def __init__(self, log_type:str="", **kwargs) -> None:
        """ Optional: log_type will be appended in the beigning of all args. """
        self._log_type = log_type
       
        self.__dict__.update(kwargs) # Add kwargs to __dict__
    
    @property
    def log_type(self) -> str:
        return self._log_type
    
    @log_type.setter
    def log_type(self, v) -> None:
        if v != self._log_type:
            self._log_type = v

    @property
    def log(self) -> dict:
        return {k:v for k, v in self.__dict__.items() if k != '_log_type'}
    
    def log_with_type(self, ignore_key:str|list|tuple|None=None) -> dict:
        if isinstance(ignore_key, str):
            ignore_key = [ignore_key]
        elif ignore_key is None:
            ignore_key = []

        log = {}
        for key, value in self.log.items():
            if key in ignore_key:
                log[key] = value
            else:
                log[f'{self._log_type} {key}'] = value
        
        return log

    def __call__(self, **kwargs) -> None:
        """
        Allows the instance to be called with additional key-value pairs to update the log.
        If a key already exists, it will be appended to a list.
        """
        for key, value in kwargs.items():
            if key in self.__dict__:
                if isinstance(self.__dict__[key], list):
                    self.__dict__[key].append(value)
                else:
                    self.__dict__[key] = [self.__dict__[key], value]
            else:
                self.__dict__[key] = value
    
    def append_log(self, new_log: Log) -> None:
        """
        Add data from new_log to this instance of the log.
        If a key already exists, append new values to a list.
        """
        new_data = new_log.log
        for key, value in new_data.items():
            if key in self.__dict__:
                if isinstance(self.__dict__[key], list):
                    self.__dict__[key].append(value)
                else:
                    self.__dict__[key] = [self.__dict__[key], value]
            else:
                self.__dict__[key] = value

    def average_all_values(self) -> Log:
        """
        Return a new Log where all numbers in a list are averaged;
        NOTE: This will ignore any nested lists and data-types that are not int or floats or tensors.
        """
        averaged_log_data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                # Calculate the average of lists containing only int or float values
                if all(isinstance(i, (int, float)) for i in value):
                    averaged_log_data[key] = sum(value) / len(value)
                # Handling nested lists is not included
            elif isinstance(value, (int, float)):
                averaged_log_data[key] = value

        return Log(**averaged_log_data)