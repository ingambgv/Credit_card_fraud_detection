import numpy as np
import pandas as pd
from scipy.stats import t

from itertools import combinations
from math import factorial

def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

def multiple_model_comparison(df_results, kfold, number_of_training_samples, rope_interval):
	
	model_scores = df_results.filter(regex=r"split\d*_test_score")
	
	n_comparisons = factorial(len(model_scores)) / (factorial(2) * factorial(len(model_scores) - 2))
	
	n_train = 2 * ( number_of_training_samples / kfold )
	n_test = ( number_of_training_samples / kfold )
	
	pairwise_t_test = []

	for model_i, model_k in combinations(range(len(model_scores)), 2):
		model_i_scores = model_scores.iloc[model_i].values
		model_k_scores = model_scores.iloc[model_k].values
		differences = model_i_scores - model_k_scores
		n = differences.shape[0]
		df = n - 1
		t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
		p_val *= n_comparisons  # implement Bonferroni correction
    					# Bonferroni can output p-values higher than 1
		p_val = 1 if p_val > 1 else p_val
		pairwise_t_test.append([model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val])

	pairwise_comp_df = pd.DataFrame(pairwise_t_test, 
					columns=["model_1", "model_2", "t_stat", "p_val"]).round(3)
	
	pairwise_bayesian = []

	for model_i, model_k in combinations(range(len(model_scores)), 2):
		model_i_scores = model_scores.iloc[model_i].values
		model_k_scores = model_scores.iloc[model_k].values
		differences = model_i_scores - model_k_scores
		t_post = t(df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test))
		worse_prob = t_post.cdf(rope_interval[0])
		better_prob = 1 - t_post.cdf(rope_interval[1])
		rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])

		pairwise_bayesian.append([worse_prob, better_prob, rope_prob])

	pairwise_bayesian_df = pd.DataFrame(pairwise_bayesian, 
					    columns=["worse_prob", "better_prob", "rope_prob"]).round(3)

	pairwise_comp_df = pairwise_comp_df.join(pairwise_bayesian_df)
	
	return(pairwise_comp_df)				
					
					
					
