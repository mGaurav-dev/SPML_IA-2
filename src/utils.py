"""Utility functions: deterministic seeding, logging, metrics calculation, and JSON persistence."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_random_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across random, numpy, and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logger(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures application logger with console and optional file handlers."""
    logger = logging.getLogger("household_energy_dp")
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to prevent duplicate lines
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculates MAE, RMSE, and R2 regression metrics in consistent physical units (kWh)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
    }


def utility_loss_percentage(base_rmse: float, dp_rmse: float) -> float:
    """Calculates relative utility degradation in RMSE compared to non-private baseline."""
    if base_rmse <= 0:
        return 0.0
    loss = ((dp_rmse - base_rmse) / base_rmse) * 100.0
    return round(float(loss), 2)


def ensure_directory(path: Path) -> Path:
    """Guarantees directory exists, creating parents if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], file_path: Path) -> Path:
    """Serializes dictionary to nicely formatted UTF-8 JSON file."""
    ensure_directory(file_path.parent)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    return file_path


def load_json(file_path: Path) -> Dict[str, Any]:
    """Loads and returns dictionary from UTF-8 JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
