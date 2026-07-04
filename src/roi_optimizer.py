"""
roi_optimizer.py
──────────────────────────────────────────────────────────────────
Budget-constrained intervention optimizer.

For each at-risk customer we compute:

    Expected Annual Revenue at Risk = Monthly Charges × 12 × risk_weight
    Expected ROI (if we intervene)  = Revenue at Risk × save_rate
    Net ROI                         = Expected ROI − cost_per_contact

Sort by Net ROI descending → take the top N that fit the budget.
This greedy ranking is optimal when customers are independent.

Risk weights (probability of churning given class):
    Loyal      → 0.02
    At-Risk    → 0.15
    Critical   → 0.50
    Will-Churn → 0.85
"""

import pandas as pd
import numpy as np

RISK_WEIGHTS = {0: 0.02, 1: 0.15, 2: 0.50, 3: 0.85}
CLASS_MAP    = {0: 'Loyal', 1: 'At-Risk', 2: 'Critical', 3: 'Will-Churn'}


def score_customers(df: pd.DataFrame, save_rate: float = 0.30) -> pd.DataFrame:
    """
    Attach ROI scoring columns to df.

    Requires:
        df['Pred_H30']       — integer class (0-3)
        df['Monthly Charges'] — numeric

    Adds:
        Risk Weight, Revenue at Risk, Expected ROI
    """
    df = df.copy()
    df['Risk Weight']     = df['Pred_H30'].map(RISK_WEIGHTS)
    mc = df['Monthly Charges'] if 'Monthly Charges' in df.columns else 60.0
    df['Revenue at Risk'] = mc * 12 * df['Risk Weight']
    df['Expected ROI']    = df['Revenue at Risk'] * save_rate
    return df


def optimize_interventions(df: pd.DataFrame,
                            budget: float,
                            cost_per_contact: float,
                            save_rate: float = 0.30):
    """
    Return the prioritised contact list and a summary dict.

    Parameters
    ----------
    df               : scored DataFrame (output of score_customers)
    budget           : total intervention budget in USD
    cost_per_contact : cost to contact one customer in USD
    save_rate        : fraction of contacted at-risk customers retained

    Returns
    -------
    contact_list : DataFrame sorted by priority (highest ROI first)
    summary      : dict with key business metrics
    """
    # Only consider at-risk customers (class >= 1)
    at_risk = df[df['Pred_H30'] >= 1].copy()
    at_risk = at_risk.sort_values('Expected ROI', ascending=False).reset_index(drop=True)

    max_contacts = int(budget / cost_per_contact)
    to_contact   = at_risk.head(max_contacts).copy()
    to_contact['Priority Rank'] = range(1, len(to_contact) + 1)
    to_contact['Net ROI']       = to_contact['Expected ROI'] - cost_per_contact
    to_contact['Cumulative Revenue Saved'] = (
        (to_contact['Revenue at Risk'] * save_rate).cumsum()
    )
    to_contact['Cumulative Spend'] = (
        to_contact['Priority Rank'] * cost_per_contact
    )

    rev_saved    = float(to_contact['Revenue at Risk'].sum() * save_rate)
    total_spend  = len(to_contact) * cost_per_contact
    net_gain     = rev_saved - total_spend
    roi_multiple = rev_saved / max(total_spend, 1)

    summary = {
        'budget'                    : budget,
        'cost_per_contact'          : cost_per_contact,
        'save_rate'                 : save_rate,
        'n_at_risk'                 : len(at_risk),
        'n_to_contact'              : len(to_contact),
        'total_intervention_cost'   : total_spend,
        'expected_customers_saved'  : int(len(to_contact) * save_rate),
        'expected_annual_rev_saved' : rev_saved,
        'net_roi'                   : net_gain,
        'roi_multiple'              : roi_multiple,
    }

    return to_contact, summary


def budget_sensitivity(df: pd.DataFrame,
                       cost_per_contact: float = 50.0,
                       save_rate: float = 0.30,
                       budget_levels: list | None = None) -> pd.DataFrame:
    """
    Run the optimizer at multiple budget levels.

    Returns a tidy DataFrame for plotting.
    """
    if budget_levels is None:
        budget_levels = [10_000, 25_000, 50_000, 75_000, 100_000,
                         150_000, 200_000, 250_000, 300_000]

    rows = []
    for b in budget_levels:
        _, s = optimize_interventions(df, b, cost_per_contact, save_rate)
        rows.append({
            'Budget'                  : b,
            'Customers Contacted'     : s['n_to_contact'],
            'Customers Saved'         : s['expected_customers_saved'],
            'Revenue Saved (Annual)'  : s['expected_annual_rev_saved'],
            'Total Spend'             : s['total_intervention_cost'],
            'Net Annual Gain'         : s['net_roi'],
            'ROI Multiple'            : round(s['roi_multiple'], 2),
        })

    return pd.DataFrame(rows)