import pandas as pd
import numpy as np
from typing import List
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from test_evaluate_functions.test_evaluate_functions import calculate_accuracies

def calculate_binary_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with the columns hours_next_four_weeks, error_next_four_weeks, error_week+1,
    error_week+2, error_week+3, error_week+4 and adds a new column called error_next_four_weeks
    that indicates if the lampost has suffered a breakdown during the future 4 weeks after current_week
    """

    dff = df.copy()

    dff["hours_next_four_weeks"] = (
        dff["hours_week+1"] +
        dff["hours_week+2"] + 
        dff["hours_week+3"] +
        dff["hours_week+4"]
    )

    # Zero rows ratio:
    print(len(dff.loc[dff["hours_next_four_weeks"] == 0]) / float(len(dff)))

    # Codify to get a categorical variables:
    dff["error_next_four_weeks"] = np.nan

    dff.loc[dff["hours_next_four_weeks"] == 0, "error_next_four_weeks"] = "No"
    dff.loc[dff["hours_next_four_weeks"] != 0, "error_next_four_weeks"] = "Yes"
    
    return dff

def split_easy_and_sudden_errors(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Returns the split between errors that occur without having more 
    that 2 previous weeks with errors and the other errors
    """
    
    dff = df.copy()
    dff["error_current_week"] = np.nan
    dff["error_week-1"] = np.nan
    dff["error_week-2"] = np.nan
    dff["error_week-3"] = np.nan
    dff["error_week-4"] = np.nan

    dff.loc[dff["hours_current_week"] == 0, "error_current_week"] = "No"
    dff.loc[dff["hours_current_week"] != 0, "error_current_week"] = "Yes"

    dff.loc[dff["hours_week-1"] == 0, "error_week-1"] = "No"
    dff.loc[dff["hours_week-1"] != 0, "error_week-1"] = "Yes"

    dff.loc[dff["hours_week-2"] == 0, "error_week-2"] = "No"
    dff.loc[dff["hours_week-2"] != 0, "error_week-2"] = "Yes"

    dff.loc[dff["hours_week-3"] == 0, "error_week-3"] = "No"
    dff.loc[dff["hours_week-3"] != 0, "error_week-3"] = "Yes"

    dff.loc[dff["hours_week-4"] == 0, "error_week-4"] = "No"
    dff.loc[dff["hours_week-4"] != 0, "error_week-4"] = "Yes"

    dff["sum_errors"] = (
        dff["error_current_week"].replace("Yes", 1).replace("No", 0) +
        dff["error_week-1"].replace("Yes", 1).replace("No", 0) +
        dff["error_week-2"].replace("Yes", 1).replace("No", 0) +
        dff["error_week-3"].replace("Yes", 1).replace("No", 0) +
        dff["error_week-4"].replace("Yes", 1).replace("No", 0)
    )
    
    # Sudden errors:
    sudden = dff.loc[(dff["sum_errors"] <= 2) & (dff["error_next_four_weeks"] == "Yes")]
    
    # Other errors:
    other_errors = dff.drop(sudden.index, axis=0)
    
    return sudden, other_errors

def split(df: pd.DataFrame, n_weeks: int) -> tuple[pd.DataFrame]:
    """
    Returns the dataframe slpitted into train, validation, test. The parameter n_weeks indicates
    the temporal split.
    """

    # The boundaries are decided deppending on the problem:
    date_test_boundary = pd.to_datetime(df["current_week"].max()) - pd.Timedelta(days=7*n_weeks+1)
    date_validation_boundary = date_test_boundary - pd.Timedelta(days=7*n_weeks+1)

    train = df.loc[df["current_week"] < date_validation_boundary].sample(frac=1.0, random_state=42)
    validation = df.loc[(df["current_week"] > date_validation_boundary) & (df["current_week"] < date_test_boundary)].sample(frac=1.0, random_state=42)
    test = df.loc[df["current_week"] >= date_test_boundary].sample(frac=1.0, random_state=42)

    print(len(train) / float(len(df)))
    print(len(validation) / float(len(df)))
    print(len(test) / float(len(df)))

    print(train["current_week"].min())
    print(train["current_week"].max())

    print(validation["current_week"].min())
    print(validation["current_week"].max())

    print(test["current_week"].min())
    print(test["current_week"].max())
    
    return train, validation, test

def split_x_y(df: pd.DataFrame, cols_to_train: List[str]) -> tuple[pd.DataFrame]:
    """
    Takes a dataframe (train, validation or test) and returns the x and y components for training and 
    calculating accuracies of the model:
    """
    return df[cols_to_train], df[["error_next_four_weeks"]]

def train_ada_boost(max_depth_tree: int, n_estimators: int, lr: int, x: pd.DataFrame, y: pd.DataFrame, random_state: int = 42) -> AdaBoostClassifier:
    """
    Takes the datasets, the parameters and the hyperparameters and returns the trained ada_boost_model
    """
    
    tree = DecisionTreeClassifier(max_depth=max_depth_tree, random_state=random_state)
    
    ada_model = AdaBoostClassifier(
        estimator=tree,
        n_estimators=n_estimators,
        learning_rate=lr,
        random_state=random_state
    )
    
    ada_model = ada_model.fit(x, y)
    
    return ada_model

def evaluate_models(max_depth_l: list, n_estimators_l: list, lr_l: list, x_train: pd.DataFrame, y_train: pd.DataFrame, x_validation: pd.DataFrame, y_validation: pd.DataFrame, prob_threshold_l: float, random_state: int = 42) -> pd.DataFrame:
    """
    Performs a search using all the parameters in the lists of hyperparameters and returns a dataframe with the
    accuracies of each one of them.
    """
    
    yy_validation = y_validation.copy()
    model_results = pd.DataFrame({"learning_rate": [], "n_estimators": [], "total_accuracy": [], "yes_accuracy": [], "no_accuracy": []})
    
    for max_depth_tree in max_depth_l:
        for n_estimators in n_estimators_l:
            for lr in lr_l:
                for prob_threshold in prob_threshold_l:
                
                    ada_model = train_ada_boost(max_depth_tree, n_estimators, lr, x_train, y_train, random_state)
                
                    prob_predictions = ada_model.predict_proba(x_validation)
                    yy_validation["No_prob"] = prob_predictions[:, 0]
                    yy_validation["Yes_prob"] = prob_predictions[:, 1]
                    yy_validation["predictions"] = np.nan
                    yy_validation.loc[yy_validation["Yes_prob"] >= prob_threshold, "predictions"] = "Yes"
                    yy_validation.loc[yy_validation["Yes_prob"] < prob_threshold, "predictions"] = "No"

                    row_to_append = calculate_accuracies(yy_validation)
                    row_to_append["prob_threshold"] = prob_threshold
                    row_to_append["n_estimators"] = n_estimators
                    row_to_append["learning_rate"] = lr
                    row_to_append["tree_depth"] = max_depth_tree

                    model_results = model_results.append(row_to_append, ignore_index=True)
    
    return model_results




