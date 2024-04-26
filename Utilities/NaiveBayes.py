################################################################################
'''
NAIVE BAYES CLASS UTILITY 
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Implement the naive bayes algorithm.
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
# NAIVE BAYES CLASS

class NaiveBayesClass:

    def __init__(self, train_fold, validation_fold, X_test, y_test):

        # drop all unnecessary columns for the algo
        train_fold[FEATURE] = train_fold[FEATURE].drop(columns=
                                         ["capital_run_length_average", 
                                          "capital_run_length_longest", 
                                          "capital_run_length_total"])
        
        validation_fold[FEATURE] = validation_fold[FEATURE].drop(columns=
                                              ["capital_run_length_average", 
                                               "capital_run_length_longest", 
                                               "capital_run_length_total"])
        
        X_test = X_test.drop(columns=["capital_run_length_average", 
                                      "capital_run_length_longest", 
                                      "capital_run_length_total"])

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
    def get_spam_type(self):
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
Ignore capitol columns
remember to use laplace smoothing (value should be more than 0)
'''
################################################################################