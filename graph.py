from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph import StateGraph, END
import operator
import pandas as pd

class AgentState(TypedDict):
    # Input
    raw_df:              Optional[Any]        # uploaded CSV as DataFrame
    # Cleaned data
    cleaned_df:          Optional[Any]
    data_quality_report: dict
    # Features
    feature_df:          Optional[Any]
    feature_cols:        List[str]
    # Predictions
    predictions_df:      Optional[Any]        # customerID + churn_prob + risk_tier
    # Explanations
    explained_df:        Optional[Any]        # + top 3 reasons per customer
    # Emails
    emails_df:           Optional[Any]        # + draft retention email
    # Meta
    error:               Optional[str]
    agent_log:           Annotated[List[str], operator.add]

from agents.data_validator   import run_data_validator
from agents.feature_engineer import run_feature_engineer
from agents.churn_predictor  import run_churn_predictor
from agents.reason_explainer import run_reason_explainer
from agents.email_drafter    import run_email_drafter

workflow = StateGraph(AgentState)

workflow.add_node("data_validator",   run_data_validator)
workflow.add_node("feature_engineer", run_feature_engineer)
workflow.add_node("churn_predictor",  run_churn_predictor)
workflow.add_node("reason_explainer", run_reason_explainer)
workflow.add_node("email_drafter",    run_email_drafter)

workflow.set_entry_point("data_validator")

def check_error(state):
    return "stop" if state.get("error") else "continue"

workflow.add_conditional_edges(
    "data_validator",
    check_error,
    {"stop": END, "continue": "feature_engineer"}
)
workflow.add_edge("feature_engineer", "churn_predictor")
workflow.add_edge("churn_predictor",  "reason_explainer")
workflow.add_edge("reason_explainer", "email_drafter")
workflow.add_edge("email_drafter",    END)

graph = workflow.compile()