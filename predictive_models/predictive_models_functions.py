import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

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

def adboc_predictor(row, ada_model, ada_sudden_model, prob_threshold_ada_model, prob_threshold_ada_sudden_model, return_probs=False):
    # Ada Boost Combined predictor
    # Gets two models and a row in the form of a pandas series and returns the prediction for that row 
    # using the models

    # Get the number of past weeks with errors
    total_week_errors = (
        row["hours_current_week"] + 
        row["hours_week-1"] + 
        row["hours_week-2"] + 
        row["hours_week-3"] +
        row["hours_week-4"]
    ).iloc[0]
    
    # This "if" checks if we want to return probabilities or we want the clasification. In the case of checking the
    # accuracies during training we will use the default value for return_probs. For returning predictions we 
    # are interested in getting probabilities to return to the user, so in this case we will set return_probs = True
    if return_probs:
        if total_week_errors <= 2:
            yes_prob = ada_sudden_model.predict_proba(row)[:, 1][0]
            return (yes_prob, "ada_sudden_model")
        if total_week_errors > 2:
            yes_prob = ada_model.predict_proba(row)[:, 1][0]
            return (yes_prob, "ada_model")
    else:
        if total_week_errors <= 2:
            yes_prob = ada_sudden_model.predict_proba(row)[:, 1]
            if yes_prob >= prob_threshold_ada_sudden_model:
                return "Yes"
            else:
                return "No"

        if total_week_errors > 2:
            yes_prob = ada_model.predict_proba(row)[:, 1]
            if yes_prob >= prob_threshold_ada_model:
                return "Yes"
            else:
                return "No"

def test_adboc_model(x, y, ada_model, ada_sudden_model, prob_threshold_ada_model, prob_threshold_ada_sudden_model):
    yy = y.copy()
    
    # Let's make the final predictions:
    predictions = []
    for _, row in x.iterrows():
        
        predictions.append(
            adboc_predictor(
                pd.DataFrame([row]), 
                ada_model, 
                ada_sudden_model, 
                prob_threshold_ada_model, 
                prob_threshold_ada_sudden_model
            )
        )

    yy["predictions"] = predictions
    
    return (calculate_accuracies(yy), yy)

def adboc_predict(df: pd.DataFrame, ada_model: AdaBoostClassifier, ada_sudden_model: AdaBoostClassifier, prob_threshold_ada_model: float, prob_threshold_ada_sudden_model: float) -> dict:

    predictions = {}

    for _, row in df.iterrows():

        predictions[row["id"]] = adboc_predictor(
            row = pd.DataFrame([row]).drop("id", axis=1),
            ada_model = ada_model,
            ada_sudden_model = ada_sudden_model,
            prob_threshold_ada_model = prob_threshold_ada_model,
            prob_threshold_ada_sudden_model = prob_threshold_ada_sudden_model,
            return_probs = True
        )
    
    return predictions