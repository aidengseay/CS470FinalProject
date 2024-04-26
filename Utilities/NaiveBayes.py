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
TOTAL = -3

################################################################################
# NAIVE BAYES CLASS

class NaiveBayesClass:

    def __init__(self, train_fold, validation_fold, X_test, y_test):

        # drop all unnecessary columns for the algo
        feature_train_fold = train_fold[FEATURE].drop(columns=
                                         ["capital_run_length_average", 
                                          "capital_run_length_longest", 
                                          "capital_run_length_total"])
        
        feature_validation_fold = validation_fold[FEATURE].drop(columns=
                                              ["capital_run_length_average", 
                                               "capital_run_length_longest", 
                                               "capital_run_length_total"])
        
        self.X_test = X_test.drop(columns=["capital_run_length_average", 
                                      "capital_run_length_longest", 
                                      "capital_run_length_total"])

        # set variables to the class struct
        self.train_fold = (feature_train_fold, train_fold[TARGET])
        self.validation_fold = (feature_validation_fold, 
                                                        validation_fold[TARGET])
        self.y_test = y_test


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
    def find_likelihoods(self, spam_count, non_spam_count):
        
        total_count = spam_count + non_spam_count

        # iterate through each row and colum
        for col in range(self.train_fold[FEATURE].shape[1]):


            for row in range(self.train_fold[FEATURE].shape[0]):
                value = self.train_fold[FEATURE].iloc[row,col]


    '''
    Parameter(s): self, ctr_code (SPAM, NOT_SPAM)
    Process: counts how many samples 
    Return: (int) the count of the sample category
    Function Dependencies: none
    '''
    def count_samples(self, ctr_code):
        if ctr_code == SPAM:
            count = self.train_fold[TARGET].value_counts()[1]
        else: # assume not spam
            count = self.train_fold[TARGET].value_counts()[0]
        return count

    '''
    Parameter(s): self
    Process:
    Return:
    Function Dependencies: count_samples
    '''
    def learn(self):
        
        # count how many samples are spam and not spam
        spam_count = self.count_samples(SPAM)
        non_spam_count = self.count_samples(NOT_SPAM)

        # calculate likelihoods
        spam_prob, non_spam_prob = self.find_likelihoods(spam_count, 
                                                                 non_spam_count)

        




################################################################################
# NOTES
'''
Ignore capitol columns
remember to use laplace smoothing (value should be more than 0)
'''
################################################################################