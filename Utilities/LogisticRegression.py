################################################################################
'''
LOGISTIC REGRESSION FUNCTION UTILITIES
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Implement the logistic regression algorithm.
'''
################################################################################
# IMPORTS


################################################################################
# CONSTANTS


################################################################################
# MAIN FUNCTION

'''
Parameter(s):
Process: Main algorithm fo logistic regression.
Return:
Function Dependencies:
'''
def logistic_regression_main():
    pass


################################################################################
# SUPPORTING FUNCTIONS


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