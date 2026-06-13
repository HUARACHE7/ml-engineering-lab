from typing import Any, Optional
from experiment_tracker.database import ExperimentDB

class MLTracker:
    def __init__(self, experiment_name: str, db_path: str = "experiments.db"):
        """
        Initializes the tracker, connects the database, and creates/gets an experiment
        """
        self.db = ExperimentDB(db_path)
        self.experiment_id = self.db.create_experiment(experiment_name)
        self.current_run_id: Optional[str] = None
        
        self._next_run_name: Optional[str] = None
        self._next_source_type: Optional[str] = None

    def start_run(self, run_name: str, source_type: str = "custom") -> "MLTracker":
        """
        Prepares the run. The method returns self to work in the 'with' block
        source_type can be 'custom' or 'sklearn'
        """
        if self.current_run_id is not None:
            raise RuntimeError("The launch is already active! Close the current launch before starting a new one.")
            
        self._next_run_name = run_name
        self._next_source_type = source_type
        return self

    def __enter__(self) -> "MLTracker":
        """
        Python magic: called automatically when entering a `with` block
        Creates a database entry with the RUNNING status
        """
        if self._next_run_name is None or self._next_source_type is None:
            raise RuntimeError("use tracker.start_run(name, type) inside the 'with' statement")
        
        self.current_run_id = self.db.create_run(
            experiment_id=self.experiment_id,
            run_name=self._next_run_name,
            source_type=self._next_source_type
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Python magic: called automatically when exiting a `with` block
        (even if an error occurred inside the block)
        """
        if self.current_run_id is None:
            return False

        if exc_type is not None:
            print(f"An error was encountered during startup: {exc_val}. Fix the FAILED status.")
            self.db.update_run_status(self.current_run_id, "FAILED")
        else:
            self.db.update_run_status(self.current_run_id, "FINISHED")

        self.current_run_id = None
        self._next_run_name = None
        self._next_source_type = None
        
        return False

    def log_param(self, key: str, value: Any):
        """Logs a hyperparameter into an active run"""
        if self.current_run_id is None:
            raise RuntimeError("No active startup! Parameter logging is not possible.")
        self.db.log_parameter(self.current_run_id, key, value)

    def log_metric(self, key: str, value: float):
        """Logs metrics into active run"""
        if self.current_run_id is None:
            raise RuntimeError("No active startup! Metrics logging is not possible.")
        self.db.log_metric(self.current_run_id, key, value)
 