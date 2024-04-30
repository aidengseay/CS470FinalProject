################################################################################
'''
NAIVE BAYES CLASS UTILITY 
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Implement the naive bayes algorithm.
'''
################################################################################
# IMPORTS
    # none

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
    Parameter(s): self, ctr_code (SPAM, NOT_SPAM)
    Process: counts how many samples of spam and not spam. 
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
    Parameter(s): self, spam_prop, non_spam_prop, spam_word_freq, 
                  non_spam_word_freq
    Process: Takes in probabilities to calculate whether or not an email is spam
             or not. 
    Return: email_result (list), true_result (list)
    Function Dependencies: none
    '''
    def evaluate(self, spam_prop, non_spam_prop, spam_word_freq, 
                                                            non_spam_word_freq):

        email_result = []
        true_result = []
        
        # iterate through the rows (emails) and cols (attributes)
        for row in range(self.validation_fold[FEATURE].shape[0]):

            spam_calc = 1
            non_spam_calc = 1

            true_result.append(self.validation_fold[TARGET].iloc[row])

            for col in range(self.validation_fold[FEATURE].shape[1]):

                value = self.validation_fold[FEATURE].iloc[row, col]

                # check if the word exists
                if value > 0.0:
                
                    spam_calc *= spam_word_freq[col]
                    non_spam_calc *= non_spam_word_freq[col]

            spam_calc *= spam_prop
            non_spam_calc *= non_spam_prop

            # identify if spam or not spam
            if spam_calc > non_spam_calc:

                # row is spam
                email_result.append(1)

            else:

                # row is not spam
                email_result.append(0)

        return email_result, true_result


    '''
    Parameter(s): self
    Process: learns with the training data. 
    Return:spam_prior_prob (int), non_spam_prior_prob (int), spam_word_freq 
           (list), non_spam_word_freq (list)
    Function Dependencies: count_samples, data_prop, word_frequency
    '''
    def learn(self):
        
        # count how many samples are spam and not spam
        spam_count = self.count_samples(SPAM)
        non_spam_count = self.count_samples(NOT_SPAM)

        # calculate likelihoods
        spam_prop, non_spam_prop = self.data_prop(spam_count, non_spam_count)

        # iterate through all features and calculate probability
        spam_word_freq, non_spam_word_freq = self.word_frequency(spam_count, 
                                                                 non_spam_count)
        
        return spam_prop, non_spam_prop, spam_word_freq, non_spam_word_freq
    
    
    '''
    Parameter(s): self, spam_count, non_spam_count
    Process: Calculates the proportion of spam and not spam
    Return: spam_prior_prob (float), non_spam_prior_prob (float)
    Function Dependencies: none
    '''
    def data_prop(self, spam_count, non_spam_count):
        
        total_count = spam_count + non_spam_count

        spam_prop = spam_count/total_count
        non_spam_prop = non_spam_count/total_count

        return spam_prop, non_spam_prop
    

    '''
    Parameter(s): self, spam_count, non_spam_count
    Process: Calculates the frequency of words showing up in spam not spam
    Return: spam_word_freq (list), non_spam_word_freq (list)
    Function Dependencies: none
    '''
    def word_frequency(self, spam_count, non_spam_count):
        non_spam_word_freq = []
        spam_word_freq = []

        # iterate through each attribute
        for col in range(self.train_fold[FEATURE].shape[1]):
            
            non_spam_word_count = 0
            spam_word_count = 0

            # iterate through each email
            for row in range(self.train_fold[FEATURE].shape[0]):

                value = self.train_fold[FEATURE].iloc[row, col]
                is_spam = self.train_fold[TARGET].iloc[row]

                # check if there is a word
                if value > 0.0:

                    if is_spam == 0: # not spam
                        non_spam_word_count += 1
                        
                    else: # is spam
                        spam_word_count += 1

            # compute the probability with laplace smoothing
            non_spam_word_count_prob = ((1 + non_spam_word_count) / 
                                                           (2 + non_spam_count))
            spam_word_count_prob = (1 + spam_word_count) / (2 + spam_count )

            # append to results list
            non_spam_word_freq.append(non_spam_word_count_prob)
            spam_word_freq.append(spam_word_count_prob)

        return spam_word_freq, non_spam_word_freq


################################################################################