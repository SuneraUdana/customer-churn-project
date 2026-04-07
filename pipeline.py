from graph import graph
import pandas as pd

def run_pipeline(df: pd.DataFrame) -> dict:
    initial_state = {
        "raw_df":              df,
        "cleaned_df":          None,
        "data_quality_report": {},
        "feature_df":          None,
        "feature_cols":        [],
        "predictions_df":      None,
        "explained_df":        None,
        "emails_df":           None,
        "error":               None,
        "agent_log":           []
    }
    result = graph.invoke(initial_state)
    return result