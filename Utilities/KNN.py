################################################################################
'''
KNN CLASS UTILITY
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Implement the k-nearest neighbors algorithm (knn).
'''
################################################################################
# IMPORTS
from sklearn.metrics import pairwise_distances
import numpy as np

################################################################################
# CONSTANTS
TARGET = 1
FEATURE = 0

################################################################################
# KNN CLASS

class KNNClass:

    def __init__(self, train_fold, validation_fold, X_test, y_test):

        # set variables to the class struct
        self.train_fold = train_fold
        self.validation_fold = validation_fold
        self.X_test = X_test
        self.y_test = y_test
        self.k = 3

    '''
    Parameter(s): self, X_dataset, y_dataset
    Process: Looks at test/validation data and makes decision based on closest
             neighbors in training data
    Return: calc_result (list), true_result (list)
    Function Dependencies: none
    '''
    def evaluate(self, X_test_dataset, y_test_dataset):

        # get the true result
        true_result = []
        for row in range(y_test_dataset.shape[0]):
            true_result.append(y_test_dataset.iloc[row])

        calc_result = []
        
        # calculate the distances
        distances = pairwise_distances(X_test_dataset, self.train_fold[FEATURE])

        # iterate through each test and find neighbors
        for test_email in distances:

            # find smallest distances in training set
            min_values_unique = list(set(test_email))
            min_values_unique = sorted(min_values_unique)
            min_values = min_values_unique[:self.k]

            # find the minimum values indices
            min_indices = []
            for value in min_values:
                min_index = np.where(test_email == value)[0]
                min_index = [index.item() for index in min_index]
                min_indices.append(min_index[0])

            # with indices find if email is spam or not
            verdicts = []
            for train_email in min_indices:
                verdicts.append(self.train_fold[TARGET].iloc[train_email])

            # Count the occurrences of spam vs non-spam
            verdicts = np.array(verdicts)
            count_spam = np.count_nonzero(verdicts == 1)
            count_non_spam = np.count_nonzero(verdicts == 0)

            if count_spam > count_non_spam:
                calc_result.append(1)
            else:
                calc_result.append(0)

        return calc_result, true_result


################################################################################