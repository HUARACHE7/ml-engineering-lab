import os
import pytest
import pandas as pd
import numpy as np

from val_framework.engine import ValidationEngine
from val_framework.models import MyLinearRegression
from val_framework.splits import custom_kfold, custom_stratified_kfold
from val_framework.evaluators import MyStandartScaler, r2_score_custom

from experiment_tracker.tracker import MLTracker


TEST_VAL_DB = "test_validation.db"


@pytest.fixture
def synthetic_regression_data():
    """
    Fixture for generating synthetic data.
    Creates a simple DataFrame with two features and a target,
    so that linear regression can learn from it.
    """
    np.random.seed(42)
    X = np.random.rand(30, 2)
    # y = 2*x1 + 3*x2 + noise
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(30) * 0.1
    
    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    df["target"] = y
    return df


@pytest.fixture
def clean_tracker():
    """
    Fixture for isolating the tracker database.
    Removes the old test database before the test and ensures cleanup afterward.
    """
    if os.path.exists(TEST_VAL_DB):
        os.remove(TEST_VAL_DB)
        
    tracker = MLTracker(experiment_name="Pytest Val Framework", db_path=TEST_VAL_DB)
    
    yield tracker
    
    if os.path.exists(TEST_VAL_DB):
        os.remove(TEST_VAL_DB)


def test_validation_engine_end_to_end(synthetic_regression_data, clean_tracker):
    """
    Integration test: verifies the complete workflow of the framework.
    Checks that folds are split correctly, scaler scales properly, model learns,
    and results are automatically saved to SQLite.
    """
    model = MyLinearRegression(eta=0.01, n_iter=100)
    scaler = MyStandartScaler()
    metrics = {"R2_score": r2_score_custom}
    
    def test_splitter(df):
        return custom_kfold(df, k=3, random_state=42)

    engine = ValidationEngine(
        model=model,
        splitter=test_splitter,
        scaler=scaler,
        metrics=metrics,
        tracker=clean_tracker
    )

    engine.run_validation(
        df=synthetic_regression_data,
        target_column="target",
        feature_columns=["feature_1", "feature_2"],
        run_name_prefix="test_lr_pipeline"
    )

    summary = clean_tracker.db.get_experiment_runs_summary("Pytest Val Framework")
    
    assert len(summary) == 3
    
    first_fold = summary[0]
    assert "test_lr_pipeline_fold" in first_fold["Name"]
    assert first_fold["Status"] == "FINISHED"
    assert first_fold["Source"] == "custom"
    
    assert "p:eta" in first_fold
    assert "m:R2_score" in first_fold
    assert float(first_fold["p:eta"]) == 0.01
    assert isinstance(first_fold["m:R2_score"], float)


@pytest.fixture
def synthetic_classification_data():
    """
    Fixture for generating classification data with strong imbalance (80% vs 20%).
    """
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    y = np.zeros(100)
    y[20:] = 1
    
    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    df["label"] = y.astype(int)
    return df


def test_stratified_kfold_proportions(synthetic_classification_data):
    """
    Tests that class proportions are preserved within each fold.
    """
    df = synthetic_classification_data
    k = 4
    
    folds = custom_stratified_kfold(df, stratify_field="label", k=k, random_state=42)
    
    assert len(folds) == k

    for train_idx, test_idx in folds:
        test_df = df.loc[test_idx]
        
        class_1_ratio = (test_df["label"] == 1).mean()
        
        assert abs(class_1_ratio - 0.8) < 0.05
