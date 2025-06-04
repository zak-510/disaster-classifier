import os
import mlflow
import torch
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """Initialize MLflow experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (default: from env var MLFLOW_TRACKING_URI)
            run_name: Optional name for this specific run
        """
        self.experiment_name = experiment_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters for the current run."""
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics for the current run."""
        mlflow.log_metrics(metrics, step=step)
        
    def log_model(self, model: torch.nn.Module, name: str) -> None:
        """Log a PyTorch model."""
        mlflow.pytorch.log_model(model, name)
        
    def log_artifact(self, local_path: str) -> None:
        """Log a local file or directory as an artifact."""
        mlflow.log_artifact(local_path)
        
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the current run."""
        mlflow.set_tags(tags)
        
    def end_run(self) -> None:
        """End the current run."""
        mlflow.end_run()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run() 