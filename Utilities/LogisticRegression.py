################################################################################
'''
LOGISTIC REGRESSION CLASS UTILITY
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Implement the logistic regression algorithm.
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
# LOGISTIC REGRESSION CLASS

class LogisticRegressionClass:

    def __init__(self, train_fold, validation_fold, X_test, y_test):

        # set variables to the class struct
        self.train_fold = train_fold
        self.validation_fold = validation_fold
        self.X_test = X_test
        self.y_test = y_test
        self.spam_samples = None
        self.email_samples = None


################################################################################
# NOTES
'''
Model learning
pred_y = Mx + b -> vector M = (m1, m2, ..., mn), b is a scalar;
loss = (pred_y - y)^2

batch gradient descent: the gradient of m is gm = sum(x*(pred_y - y))*2/n, and 
the gradient of b is gb = sum(pred_y - y)*2/n where n is the size of the batch

performing gradient descent
for i in range(epoch):
    pred_y = ...
    gm = ...
    gd = ...
    m = m - learning rate THERE IS MORE
'''
################################################################################