import numpy as np
import pandas as pd
from typing import Any, List, Dict, Callable
from contextlib import nullcontext
from experiment_tracker.tracker import MLTracker

class ValidationEngine:
    def __init__(
        self,
        model: Any,
        splitter: Callable[[pd.DataFrame], List[tuple]],
        scaler: Any = None,
        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
        tracker: MLTracker = None
    ):
        """
        Universal engine for model validation.
        """
        self.model = model
        self.splitter = splitter
        self.scaler = scaler
        self.metrics = metrics or {}
        self.tracker = tracker

    def run_validation(self, df: pd.DataFrame, target_column: str, feature_columns: List[str], run_name_prefix: str):
        """
        Runs a complete cross-validation cycle across all folds.
        """
        folds = self.splitter(df)
        fold_metrics = {metric_name: [] for metric_name in self.metrics.keys()}
        
        print(f"Starting cross-validation for model {self.model.__class__.__name__} ({len(folds)} folds)...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            run_name = f"{run_name_prefix}_fold_{fold_idx + 1}"
            
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]
            
            X_train, y_train = df_train[feature_columns].values, df_train[target_column].values
            X_test, y_test = df_test[feature_columns].values, df_test[target_column].values
            
            context = self.tracker.start_run(run_name=run_name, source_type="custom") if self.tracker else nullcontext()
            
            with context:
                if self.tracker:
                    if hasattr(self.model, "eta"): self.tracker.log_param("eta", self.model.eta)
                    if hasattr(self.model, "n_iter"): self.tracker.log_param("n_iter", self.model.n_iter)
                    if hasattr(self.model, "n_neighbors"): self.tracker.log_param("n_neighbors", self.model.n_neighbors)
                
                if self.scaler:
                    self.scaler.fit(X_train)
                    X_train = self.scaler.transform(X_train)
                    X_test = self.scaler.transform(X_test)
                
                self.model.fit(X_train, y_train)
                
                y_pred = self.model.predict(X_test)
                
                for metric_name, metric_fn in self.metrics.items():
                    score = metric_fn(y_test, y_pred)
                    fold_metrics[metric_name].append(score)
                    
                    print(f"  Fold {fold_idx + 1} | {metric_name}: {score:.4f}")
                    
                    if self.tracker:
                        self.tracker.log_metric(metric_name, score)
                        
        for metric_name, scores in fold_metrics.items():
            mean_score = np.mean(scores)
            print(f"Mean {metric_name}: {mean_score:.4f} (std: {np.std(scores):.4f})")