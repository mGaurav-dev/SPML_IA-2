"""Data loading, schema adaptation, chunked reading, and smoke dataset generation for IA-2."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ProjectPaths

logger = logging.getLogger("household_energy_dp")


# Canonical schema dictionary mapping canonical names to known alias patterns
SCHEMA_ALIASES = {
    "household_id": ["lclid", "household_id", "customer_id", "id", "meter_id", "household"],
    "timestamp": ["datetime", "timestamp", "date", "date_time", "local_time", "time"],
    "energy_kwh": [
        "kwh/hh (per half hour)",
        "kwh/hh (per half hour) ",
        "energy_kwh",
        "kwh",
        "consumption_kwh",
        "usage_kwh",
        "energy",
        "consumption",
    ],
}


def _clean_column_name(col: str) -> str:
    """Strips whitespace and non-alphanumeric characters for robust schema matching."""
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def resolve_dataset_path(
    user_path: Optional[str | Path] = None,
    paths: Optional[ProjectPaths] = None,
) -> Optional[Path]:
    """Resolves the dataset path by inspecting candidate locations in order of priority.

    Priority:
        1. Explicit user_path
        2. IA-2 local raw data folder: E:\\SPML\\IA2\\data\\raw\\LCL-June2015v2_0.csv
        3. ProjectPaths data_raw
        4. Parent IA-1 workspace (read-only fallback): E:\\SPML\\IA1\\LCL-June2015v2_0.csv

    Returns:
        Resolved Path object if a valid CSV is found, otherwise None.
    """
    if user_path:
        p = Path(user_path)
        if p.is_file() and p.suffix.lower() == ".csv":
            return p.resolve()
        if p.is_dir():
            for child in sorted(p.glob("*.csv")):
                if child.is_file():
                    return child.resolve()
        return None

    candidates: List[Path] = []
    if paths is not None:
        candidates.extend([
            paths.data_raw / "LCL-June2015v2_0.csv",
            paths.data_raw,
        ])

    # Direct workspace paths
    candidates.extend([
        Path("e:/SPML/IA2/data/raw/LCL-June2015v2_0.csv"),
        Path("data/raw/LCL-June2015v2_0.csv"),
        Path("e:/SPML/IA1/LCL-June2015v2_0.csv"),  # Read-only fallback
        Path("../IA1/LCL-June2015v2_0.csv"),
    ])

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".csv":
            return candidate.resolve()
        if candidate.is_dir():
            for child in sorted(candidate.glob("*.csv")):
                if child.is_file():
                    return child.resolve()

    return None


def adapt_schema(
    df: pd.DataFrame,
    required_canonical: Sequence[str] = ("household_id", "timestamp", "energy_kwh"),
) -> pd.DataFrame:
    """Adapts raw source column names to canonical internal schema names.

    Args:
        df: Raw DataFrame chunk.
        required_canonical: Canonical names that must be resolved.

    Returns:
        DataFrame with normalized canonical column names.
    """
    column_mapping = {}
    normalized_cols = {col: _clean_column_name(col) for col in df.columns}

    for canonical, aliases in SCHEMA_ALIASES.items():
        cleaned_aliases = [_clean_column_name(a) for a in aliases]
        for orig_col, norm_col in normalized_cols.items():
            if norm_col in cleaned_aliases:
                column_mapping[orig_col] = canonical
                break

    missing = [c for c in required_canonical if c not in column_mapping.values()]
    if missing:
        raise ValueError(
            f"Missing required canonical fields: {missing}. "
            f"Available raw columns: {list(df.columns)}. "
            f"Resolved mapping: {column_mapping}"
        )

    renamed = df.rename(columns=column_mapping)
    columns_to_keep = [c for c in required_canonical if c in renamed.columns]
    result = renamed[columns_to_keep].copy()

    # Convert timestamp to datetime
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    # Convert energy_kwh to numeric, replacing 'Null' with NaN
    result["energy_kwh"] = pd.to_numeric(result["energy_kwh"], errors="coerce")

    # Strip whitespace from household_id string
    result["household_id"] = result["household_id"].astype(str).str.strip()

    return result


def load_dataset(
    file_path: Path,
    target_household: Optional[str] = "MAC000002",
    max_rows: Optional[int] = 25000,
    chunk_size: int = 50000,
) -> Tuple[pd.DataFrame, dict]:
    """Loads dataset from CSV in chunked streaming fashion with memory-safety guarantees.

    Args:
        file_path: Resolved path to CSV file.
        target_household: Anonymized household identifier to filter (e.g. 'MAC000002').
        max_rows: Maximum rows to retain. If None, retains all matching rows.
        chunk_size: Number of rows read per chunk.

    Returns:
        Tuple of (combined DataFrame, dictionary of loading statistics).
    """
    logger.info("Streaming dataset from %s (target_household=%s, chunk_size=%d)",
                file_path, target_household, chunk_size)

    matching_chunks: List[pd.DataFrame] = []
    total_scanned_rows = 0
    total_matching_rows = 0

    reader = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_scanned_rows += len(chunk)
        adapted_chunk = adapt_schema(chunk)

        if target_household is not None:
            filtered = adapted_chunk[adapted_chunk["household_id"] == target_household].copy()
        else:
            filtered = adapted_chunk

        if not filtered.empty:
            matching_chunks.append(filtered)
            total_matching_rows += len(filtered)

        if max_rows is not None and total_matching_rows >= max_rows:
            logger.info("Reached target row count threshold (%d rows matching). Stopping stream.",
                        total_matching_rows)
            break

    if not matching_chunks:
        raise ValueError(
            f"No records found for target_household='{target_household}' in {file_path}. "
            f"Scanned {total_scanned_rows} total rows."
        )

    df_combined = pd.concat(matching_chunks, ignore_index=True)
    if max_rows is not None and len(df_combined) > max_rows:
        df_combined = df_combined.iloc[:max_rows].copy()

    stats = {
        "file_path": str(file_path),
        "target_household": target_household,
        "scanned_rows": total_scanned_rows,
        "loaded_matching_rows": len(df_combined),
    }

    logger.info("Loaded %d rows for household '%s' from %d scanned rows.",
                len(df_combined), target_household, total_scanned_rows)
    return df_combined, stats


def generate_synthetic_smoke_data(
    n_rows: int = 1500,
    household_id: str = "SYNTHETIC_HH_01",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """Generates synthetic half-hourly time series data for smoke testing."""
    rng = np.random.default_rng(random_seed)
    start_time = pd.Timestamp("2013-01-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(minutes=30 * i) for i in range(n_rows)]

    # Diurnal pattern + weekly variation + noise
    hours = np.array([t.hour + t.minute / 60.0 for t in timestamps])
    day_of_week = np.array([t.dayofweek for t in timestamps])

    # Base morning peak (8am) and evening peak (8pm)
    morning_peak = 0.5 * np.exp(-((hours - 8.0) ** 2) / 4.0)
    evening_peak = 0.9 * np.exp(-((hours - 20.0) ** 2) / 6.0)
    weekend_boost = np.where(day_of_week >= 5, 0.2, 0.0)
    noise = rng.exponential(scale=0.1, size=n_rows)

    energy_kwh = np.clip(0.1 + morning_peak + evening_peak + weekend_boost + noise, 0.0, 4.0)

    # Inject minor synthetic anomalies
    if n_rows > 100:
        energy_kwh[25] = np.nan  # Short missing gap
        energy_kwh[50] = np.nan

    df = pd.DataFrame({
        "household_id": household_id,
        "timestamp": timestamps,
        "energy_kwh": energy_kwh,
    })

    stats = {
        "file_path": "SYNTHETIC_IN_MEMORY",
        "target_household": household_id,
        "scanned_rows": n_rows,
        "loaded_matching_rows": n_rows,
    }
    return df, stats
