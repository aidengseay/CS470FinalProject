################################################################################
'''
ANALYSIS FUNCTION UTILITIES
Credit: Aiden Seay - Spring 2024

DESCRIPTION:
Performs analysis on ML algorithms going over acc, fp, tp, and AUC.
'''
################################################################################
# IMPORTS
from sklearn.metrics import roc_auc_score
from statistics import mean

################################################################################
# CONSTANTS
    # none

################################################################################
# MAIN FUNCTIONS

'''
Parameter(s): calc_result (list), true_result (list)
Process: performs an analysis on the algorithm
Return: acc, fp, tp, and AUC
Function Dependencies: find_acc, find_auc, find_false_pos, find_true_pos
'''
def analyze_results(calc_result, true_result):
    
    # calculate the accuracy
    acc = find_acc(calc_result, true_result)

    # calculate false positive
    fp = find_false_pos(calc_result, true_result)

    # calculate true positive
    tp = find_true_pos(calc_result, true_result)

    # calculate AUC
    auc = find_auc(calc_result, true_result)

    return acc, fp, tp, auc

'''
Parameter(s): acc_list (list), fp_list (list), tp_list (list), auc_list (list)
Process: averages all the results from each fold
Return: acc_avg (float), fp_avg (float), tp_avg (float), auc_avg (float)
Function Dependencies: mean
'''
def average_stats(acc_list, fp_list, tp_list, auc_list):
    
    acc_avg = mean(acc_list)
    fp_avg = mean(fp_list)
    tp_avg = mean(tp_list)
    auc_avg = mean(auc_list)

    return acc_avg, fp_avg, tp_avg, auc_avg


################################################################################
# SUPPORTING FUNCTIONS

'''
Parameter(s): calc_result (list), true_result (list)
Process: Calculates the accuracy rate for the calculated result and true result
Return: accuracy (float)
Function Dependencies: none
'''
def find_acc(calc_result, true_result):
    
    # get all correct guesses
    correct_count = sum(1 for i, j in zip(calc_result, true_result) if i == j)
    total_questions = len(true_result)
    return correct_count / total_questions


'''
Parameter(s): calc_result (list), true_result (list)
Process: Calculates the area under the Receiver Operating Characteristic 
         (ROC) curve
Return: auc (float)
Function Dependencies: roc_auc_score
'''
def find_auc(calc_result, true_result):
    return roc_auc_score(true_result, calc_result)


'''
Parameter(s): calc_result (list), true_result (list)
Process: Calculates the false positive rate (non spam incorrectly labeled spam)
Return: false_pos_rate (float)
Function Dependencies: none
'''
def find_false_pos(calc_result, true_result):

    # get all false positives
    false_positive_count = sum(1 for i, j in zip(calc_result, 
                                          true_result) if i == 1 and j == 0)
    
    # get all true negatives
    true_negative_count = sum(1 for i, j in zip(calc_result, 
                                          true_result) if i == 0 and j == 0)
    
    # calculate false positive rate
    return false_positive_count / true_negative_count


'''
Parameter(s): calc_result (list), true_result (list)
Process: Calculates the true positive rate (spam correctly labeled spam)
Return: true_pos_rate (float)
Function Dependencies: none
'''
def find_true_pos(calc_result, true_result):
    
    # get all true positives
    true_positive_count = sum(1 for i, j in zip(calc_result,
                                          true_result) if i == 1 and j == 1)
    
    # get all positives
    positive_count = sum(1 for j in true_result if j == 1)

    # calculate the true pos rate
    return true_positive_count / positive_count


################################################################################