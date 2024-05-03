################################################################################
'''
LOGISTIC REGRESSION CLASS UTILITY
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Implement the logistic regression algorithm.
'''
################################################################################
# IMPORTS
import numpy as np
from Utilities.Analysis import find_acc

################################################################################
# CONSTANTS
TARGET = 1
FEATURE = 0

################################################################################
# LOGISTIC REGRESSION CLASS

class LogisticRegressionClass:

    def __init__(self, train_fold, validation_fold, X_test, y_test):

        # add column of 1's to all feature sets
        train_feature_fold = train_fold[FEATURE].copy()
        train_feature_fold["one_const"] = 1

        validation_feature_fold = validation_fold[FEATURE].copy()
        validation_feature_fold["one_const"] = 1

        X_test_new_col = X_test.copy()
        X_test_new_col["one_const"] = 1
        
        # set variables to the class struct and convert to numpy array
        self.train_fold = (train_feature_fold.to_numpy(), 
                           train_fold[TARGET].to_numpy())
        
        self.validation_fold = (validation_feature_fold.to_numpy(), 
                                validation_fold[TARGET].to_numpy())
        
        self.X_test = X_test_new_col.to_numpy()
        self.y_test = y_test.to_numpy()
        self.M = None
        self.epoc = 1500
        self.learning_rate = 0.01


    '''
    Parameter(s): self, test_dataset (numpy), answer_dataset (numpy)
    Process: Calculates the predicted results and determines if email is spam or
             not.
    Return: algo_result (list), true_result (list)
    Function Dependencies: none
    '''
    def evaluate(self, test_dataset, answer_dataset):
        
        # get predicted y results
        pred_y = (self.sigmoid(np.dot(test_dataset, self.M)))

        # get expected results
        true_result = []
        for verdict in answer_dataset:
            true_result.append(verdict)

        # get logistic regression results
        algo_result = []
        for verdict in pred_y:
                if verdict > 0.5:
                    algo_result.append(1)
                else:
                    algo_result.append(0)

        return algo_result, true_result


    '''
    Parameter(s): self
    Process: calculate the gradient descent and find the best model.
    Return: none (set self.M to best model)
    Function Dependencies: find_acc
    '''
    def gradient_descent(self):
        
        # initialize best model
        best_model = self.M
        best_performance = 0

        for shift in range(self.epoc):

            # make predictions
            pred_y = (self.sigmoid(np.dot(self.train_fold[FEATURE], self.M)))

            # calculate the gradient
            gm = np.dot(self.train_fold[FEATURE].T , 
                        (pred_y - (self.train_fold[TARGET].reshape(-1,1))) * 2 
                                            / self.train_fold[TARGET].shape[0])

            # calculate the performance
            calc_result, true_result = self.evaluate(
                    self.validation_fold[FEATURE], self.validation_fold[TARGET])
            acc = find_acc(calc_result, true_result)

            # check for best performance
            if acc > best_performance:
                best_model = self.M
                best_performance = acc

            self.M = self.M - self.learning_rate * gm

        self.M = best_model


    '''
    Parameter(s): self
    Process: initialize a random vector M and shift vector to fit training
             points with gradient descent
    Return: none
    Function Dependencies: gradient_descent
    '''
    def learn(self):

        # initialize random vector M
        self.M = np.random.randn(self.train_fold[FEATURE].shape[1], 1)

        # perform gradient descent
        self.gradient_descent()


    '''
    Parameter(s): self, x (numpy)
    Process: return sigmoid equation results
    Return: sigmoid result (numpy)
    Function Dependencies: none
    '''
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


################################################################################