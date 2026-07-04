"""
temporal_features.py
──────────────────────────────────────────────────────────────────
Simulate how customer features evolve across 30 / 60 / 90-day
horizons and classify each customer's risk trajectory.

Approach: XGBoost learned that customers with different tenure /
charge profiles carry different churn risk.  By projecting
time-sensitive features forward we can ask: "What risk tier
will this customer fall into in N months?"

Features evolved:
  Tenure Months  → +N months
  Total Charges  → + Monthly Charges × N  (spending accumulates)
  Charge/Tenure  → recalculated (ratio changes as tenure grows)
  Revenue Share  → recalculated

All other features (contract type, services, payment method)
are held constant — they don't change unless a customer actively
makes a change.
"""

import pandas as pd
import numpy as np

CLASS_MAP = {0: 'Loyal', 1: 'At-Risk', 2: 'Critical', 3: 'Will-Churn'}


def perturb_features(df_X: pd.DataFrame,
                     horizon_months: int,
                     features_list: list) -> pd.DataFrame:
    """
    Return a copy of df_X with time-sensitive features projected
    horizon_months into the future.

    Parameters
    ----------
    df_X          : preprocessed feature DataFrame (encoded)
    horizon_months: 0 = current, 2 = +60 days, 3 = +90 days
    features_list : list of column names to keep (model FEATURES)

    Returns
    -------
    DataFrame with same columns as df_X[features_list]
    """
    df_h = df_X.copy()

    if horizon_months == 0:
        return df_h[features_list]

    # --- Evolve tenure ----------------------------------------
    if 'Tenure Months' in df_h.columns:
        df_h['Tenure Months'] = df_h['Tenure Months'] + horizon_months

    # --- Accumulate total charges ----------------------------
    if 'Total Charges' in df_h.columns and 'Monthly Charges' in df_h.columns:
        df_h['Total Charges'] = (
            df_h['Total Charges'] + df_h['Monthly Charges'] * horizon_months
        )

    # --- Recalculate derived features ------------------------
    if 'Monthly Charges' in df_h.columns and 'Tenure Months' in df_h.columns:
        df_h['Charge per Tenure'] = (
            df_h['Monthly Charges'] / (df_h['Tenure Months'] + 1)
        )

    if 'Monthly Charges' in df_h.columns:
        df_h['Revenue Share'] = (
            df_h['Monthly Charges'] / df_h['Monthly Charges'].max()
        )

    return df_h[features_list]


def classify_trajectory(h30: int, h60: int, h90: int) -> str:
    """
    Classify a customer's risk trajectory across three horizons.

    Returns one of:
        'Stable'           — same class at 30d and 90d
        'Improving'        — risk decreasing
        'Escalating'       — risk increasing by 1 class
        'Rapid Escalation' — risk increasing by 2+ classes
    """
    delta = int(h90) - int(h30)

    if delta >= 2:
        return 'Rapid Escalation'
    elif delta == 1:
        return 'Escalating'
    elif delta == 0:
        return 'Stable'
    else:
        return 'Improving'


def add_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Trajectory and Risk_Delta columns to a DataFrame that
    already has Pred_H30, Pred_H60, Pred_H90 integer columns.
    """
    df = df.copy()
    df['Trajectory'] = df.apply(
        lambda r: classify_trajectory(r['Pred_H30'], r['Pred_H60'], r['Pred_H90']),
        axis=1
    )
    df['Risk_Delta'] = df['Pred_H90'].astype(int) - df['Pred_H30'].astype(int)
    return df