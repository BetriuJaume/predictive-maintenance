import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from typing import List
import pickle

def calculate_accuracies(y: pd.DataFrame) -> dict:
    """
    Takes as input a dataframe with two columns. The first one with the real values and
    the second one with the predictions and returns the accuracies of the predictions
    """

    total_accuracy = len(y.loc[y["error_next_four_weeks"] == y["predictions"]]) / float(len(y))
    # Percentage of Yes detected
    yes_accuracy = len(y.loc[(y["error_next_four_weeks"] == "Yes") & (y["predictions"] == "Yes")]) / float(len(y.loc[(y["error_next_four_weeks"] == "Yes")]))
    # Same with No:
    if len(y.loc[(y["error_next_four_weeks"] == "No")]) != 0: # In case we get a zero division
        no_accuracy = len(y.loc[(y["error_next_four_weeks"] == "No") & (y["predictions"] == "No")]) / float(len(y.loc[(y["error_next_four_weeks"] == "No")]))
    else:
        no_accuracy = None
    return {"total_accuracy": total_accuracy, "yes_accuracy": yes_accuracy, "no_accuracy": no_accuracy}

def variable_importance(ada_model: AdaBoostClassifier, trained_columns: List[int]) -> pd.DataFrame:
    """
    Takes the pretrained model and a list of the columns used to train the model and returns a dataframe 
    with the importance of all the variables in the model
    """
    
    variable_importance_df = pd.DataFrame({"variable": [], "importance": []})
    for i, importance in enumerate(ada_model.feature_importances_):
        row_to_append = {"variable": trained_columns[i], "importance": importance}
        variable_importance_df = variable_importance_df.append(row_to_append, ignore_index=True)
    
    return variable_importance_df

def return_variable_importance(ada_model: AdaBoostClassifier, trained_columns: List[int]) -> pd.DataFrame:
    """
    The function variable_importances is deprecated because of the append function of the dataframe. Use this one instead.
    Takes the pretrained model and a list of the columns used to train the model and returns a dataframe 
    with the importance of all the variables in the model
    """
    
    variable_importance_df = pd.DataFrame({"variable": [], "importance": []})
    for i, importance in enumerate(ada_model.feature_importances_):
        row_to_append = {"variable": trained_columns[i], "importance": importance}
        variable_importance_df = pd.concat(
            [
                variable_importance_df,
                pd.DataFrame([row_to_append])
            ]
        )
    
    return variable_importance_df

def test_model(model: AdaBoostClassifier, x: pd.DataFrame, y: pd.DataFrame, prob_threshold: float = 0.5) -> dict:
    """
    Takes the pretrained model, the x and y dataframes and returns the accuracies of the predictions
    """
    yy = y.copy()
    
    prob_predictions = model.predict_proba(x)
    yy["No_prob"] = prob_predictions[:, 0]
    yy["Yes_prob"] = prob_predictions[:, 1]
    yy["predictions"] = np.nan
    yy.loc[yy["Yes_prob"] >= prob_threshold, "predictions"] = "Yes"
    yy.loc[yy["Yes_prob"] < prob_threshold, "predictions"] = "No"
    
    return calculate_accuracies(yy)