################################################################################
'''
KNN CLASS UTILITY
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Implement the k-nearest neighbors algorithm (knn).
'''
################################################################################
# IMPORTS


################################################################################
# CONSTANTS
TARGET = 1
FEATURE = 0
SPAM = -1
NOT_SPAM = -2

################################################################################
# KNN CLASS

class KNNClass:

    def __init__(self, train_fold, validation_fold, X_test, y_test):

        # set variables to the class struct
        self.train_fold = train_fold
        self.validation_fold = validation_fold
        self.X_test = X_test
        self.y_test = y_test
        self.spam_samples = None
        self.email_samples = None
    
    
    '''
    Parameter(s):
    Process:
    Return:
    Function Dependencies:
    '''
    def evaluate(self):
        pass


    '''
    Parameter(s):
    Process:
    Return:
    Function Dependencies:
    '''
    def learn(self):
        pass


################################################################################
# NOTES
'''
Include all features
parallel programming
'''
################################################################################