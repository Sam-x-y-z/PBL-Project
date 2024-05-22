import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr

#Spearman Correlation

def spearman_correlation(predicted, true):
    spearman_correlation, p_value = spearmanr(predicted, true)
    return spearman_correlation, p_value

def KendallTau(predicted, true):
    kendall_tau, p_value = kendalltau(predicted, true)
    return kendall_tau, p_value

def RMSE(predicted, true):
    n = predicted.size(0)
    RMSE = torch.sqrt(torch.sum((predicted - true) ** 2) / n)
    return RMSE

def PearsonCorrelation(predicted, true):
    pearson_correlation, p_value = pearsonr(predicted, true)
    return pearson_correlation, p_value