import os
import pytest

from experiment_tracker.tracker import MLTracker
from experiment_tracker.database import ExperimentDB

TEST_DB_PATH = "test_experiments.db"

@pytest.fixture
def clean_db():
    """
    A fixture for creating an isolated testing environment.
    Before a test, it deletes the old test database, if any remains.
    After the test, it deletes the temporary file.
    """
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        
    yield TEST_DB_PATH
    
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


def test_experiment_creation_and_uniqueness(clean_db):
    """Check that experiments are created correctly and that names are unique."""
    db = ExperimentDB(clean_db)
    
    exp_id_1 = db.create_experiment("Test Exp")
    assert exp_id_1 == 1
    
    exp_id_2 = db.create_experiment("Test Exp")
    assert exp_id_1 == exp_id_2
    
    exp_id_3 = db.create_experiment("Another Exp")
    assert exp_id_3 == 2


def test_tracker_successful_run(clean_db):
    """We're testing the ideal tracker scenario in the context manager."""
    tracker = MLTracker(experiment_name="Linear Benchmark", db_path=clean_db)
    
    with tracker.start_run(run_name="custom_lr", source_type="custom") as run:
        run.log_param("learning_rate", 0.01)
        run.log_metric("R2", 0.85)
        
    db = ExperimentDB(clean_db)
    summary = db.get_experiment_runs_summary("Linear Benchmark")
    
    assert len(summary) == 1
    run_data = summary[0]
    
    assert run_data["Name"] == "custom_lr"
    assert run_data["Status"] == "FINISHED"
    assert run_data["Source"] == "custom"
    assert run_data["p:learning_rate"] == "0.01"
    assert run_data["m:R2"] == 0.85


def test_tracker_failed_run(clean_db):
    """We check that when the code crashes, the launch status automatically changes to FAILED."""
    tracker = MLTracker(experiment_name="Linear Benchmark", db_path=clean_db)
    
    try:
        with tracker.start_run(run_name="broken_lr", source_type="sklearn") as run:
            run.log_param("n_estimators", 100)
            raise ZeroDivisionError("Critical algorithm failure")
    except ZeroDivisionError:
        pass
        
    db = ExperimentDB(clean_db)
    summary = db.get_experiment_runs_summary("Linear Benchmark")
    
    assert len(summary) == 1
    run_data = summary[0]
    
    assert run_data["Name"] == "broken_lr"
    assert run_data["Status"] == "FAILED"
    assert run_data["p:n_estimators"] == "100"


def test_tracker_logging_without_active_run(clean_db):
    """We check the protection: it is impossible to log parameters without calling start_run()."""
    tracker = MLTracker(experiment_name="Security Test", db_path=clean_db)
    
    with pytest.raises(RuntimeError):
        tracker.log_param("test", 123)
