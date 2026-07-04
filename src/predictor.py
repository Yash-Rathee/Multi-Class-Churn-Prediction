"""
predictor.py
──────────────────────────────────────────────────────────────────
Shared preprocessing pipeline and prediction interface used by
both the Phase 2 notebook and the Streamlit dashboard.

Replicates Phase 1 preprocessing exactly so that new CSV uploads
in the dashboard receive identical treatment.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

from .temporal_features import perturb_features, add_trajectories

CLASS_MAP = {0: 'Loyal', 1: 'At-Risk', 2: 'Critical', 3: 'Will-Churn'}

LEAKAGE_COLS = [
    'Churn Label', 'Churn Value', 'Churn Score',
    'Churn Reason', 'Churn Class Name', 'Churn Class',
]
GEO_COLS = [
    'CustomerID', 'Count', 'Country', 'State', 'City',
    'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
]
SERVICE_COLS = [
    'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies',
]


def load_artifacts(outputs_dir: str = 'outputs') -> dict:
    """Load all saved model artifacts from disk."""
    artifacts = {}
    keys = ['model_h30', 'model_h60', 'model_h90',
            'label_encoders', 'feature_names']
    for key in keys:
        path = os.path.join(outputs_dir, f'{key}.pkl')
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
    return artifacts


def preprocess_data(df_raw: pd.DataFrame,
                    label_encoders: dict | None = None,
                    fit_encoders: bool = False):
    """
    Apply the same preprocessing pipeline as Phase 1.

    Parameters
    ----------
    df_raw         : raw Telco DataFrame
    label_encoders : dict of {col: LabelEncoder} for inference;
                     None to fit new encoders (training mode)
    fit_encoders   : if True, fit and return new encoders

    Returns
    -------
    df             : processed feature DataFrame (no target columns)
    label_encoders : (only when fit_encoders=True)
    """
    df = df_raw.copy()

    # Fix Total Charges (sometimes stored as string with spaces)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'].fillna(df.get('Monthly Charges', pd.Series(0)), inplace=True)

    # Drop leakage, geo, and any leftover cluster columns
    extra = ['Segment', 'Segment Name', 'Tenure Group', '_sc']
    drop_cols = [c for c in LEAKAGE_COLS + GEO_COLS + extra if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # ── Engineered features ──────────────────────────────────
    df['Charge per Tenure'] = (
        df['Monthly Charges'] / (df['Tenure Months'] + 1)
    )
    df['Revenue Share'] = (
        df['Monthly Charges'] / df['Monthly Charges'].max()
    )
    svc = [c for c in SERVICE_COLS if c in df.columns]
    df['Service Count'] = (df[svc] == 'Yes').sum(axis=1)

    # ── Encode categoricals ──────────────────────────────────
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if fit_encoders:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        return df, label_encoders

    elif label_encoders:
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: int(le.transform([x])[0])
                    if x in le.classes_ else -1
                )
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    else:
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def make_temporal_predictions(df_model: pd.DataFrame,
                               artifacts: dict,
                               feature_names: list) -> pd.DataFrame:
    """
    Generate H30, H60, H90 predictions using feature perturbation.

    Uses the *same* trained model with projected feature sets for
    each horizon — the model outputs different predictions because
    the input features differ (longer tenure, higher total charges).

    Parameters
    ----------
    df_model      : preprocessed DataFrame (from preprocess_data)
    artifacts     : dict containing model_h30 (and optionally h60/h90)
    feature_names : ordered list of feature columns

    Returns
    -------
    DataFrame with prediction columns + trajectories
    """
    model = artifacts['model_h30']   # same model for all horizons

    X_h30 = perturb_features(df_model, horizon_months=0, features_list=feature_names)
    X_h60 = perturb_features(df_model, horizon_months=2, features_list=feature_names)
    X_h90 = perturb_features(df_model, horizon_months=3, features_list=feature_names)

    pred_h30 = model.predict(X_h30)
    pred_h60 = model.predict(X_h60)
    pred_h90 = model.predict(X_h90)
    proba_h30 = model.predict_proba(X_h30)

    result = pd.DataFrame({
        'Pred_H30'      : pred_h30,
        'Pred_H60'      : pred_h60,
        'Pred_H90'      : pred_h90,
        'Pred_H30_Name' : [CLASS_MAP[p] for p in pred_h30],
        'Pred_H60_Name' : [CLASS_MAP[p] for p in pred_h60],
        'Pred_H90_Name' : [CLASS_MAP[p] for p in pred_h90],
        'P_Loyal'       : proba_h30[:, 0],
        'P_AtRisk'      : proba_h30[:, 1],
        'P_Critical'    : proba_h30[:, 2],
        'P_WillChurn'   : proba_h30[:, 3],
    })

    return add_trajectories(result)