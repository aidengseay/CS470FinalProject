################################################################################
'''
RETRIEVE/SPLIT DATASET FUNCTION UTILITIES
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Splits dataset to training (80%), and test (20%). Inside the training dataset
complete a 5-cross validation (80% training and 20% evaluation).
'''
################################################################################
# IMPORTS
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

################################################################################
# CONSTANTS
    # NONE

################################################################################
# SUPPORTING FUNCTIONS

'''
Parameter(s): file_path (str)
Process: Reads file and adds data to a pandas data frame.
Return: pandas data frame
Function Dependencies: None
'''
def read_dataset(file_path):
    return pd.read_csv(file_path)


'''
Parameter(s): data (pandas df)
Process: Splits dataset to training (80%), and test (20%). Inside the training 
         dataset complete a 5-cross validation (80% training and 20% 
         evaluation).
Return: train_data (list), validation_data (list), X_test (list), y_test (list)
Function Dependencies: None
'''
def split_dataset(df):

    # split the dataset into features (X) and target variable (y)
    X = df.drop(columns=["spam"])
    y = df["spam"]
    
    # split dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                                random_state=24)

    # 5 cross validation on training set
    kf = KFold(n_splits=5, shuffle=True, random_state=24)
    train_data = []
    validation_data = []

    for train_index, validation_index in kf.split(X_train):
        X_train_fold, X_validation_fold = X_train.iloc[train_index], X_train.iloc[validation_index]
        y_train_fold, y_validation_fold = y_train.iloc[train_index], y_train.iloc[validation_index]
        train_data.append((X_train_fold, y_train_fold))
        validation_data.append((X_validation_fold, y_validation_fold))

    return train_data, validation_data, X_test, y_test


################################################################################