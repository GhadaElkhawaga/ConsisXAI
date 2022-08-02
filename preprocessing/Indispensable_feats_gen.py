import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from skrebate import TuRF


def compute_impurity(feature, impurity_criterion):
    """
    This function calculates impurity of a feature.
    Supported impurity criteria: 'entropy', 'gini'
    input: feature (this needs to be a Pandas series)
    output: feature impurity
    """
    probs = feature.value_counts(normalize=True)
    if impurity_criterion == 'entropy':
        impurity = -1 * np.sum(np.log2(probs) * probs)
    elif impurity_criterion == 'gini':
        impurity = 1 - np.sum(np.square(probs))
    else:
        raise ValueError('Unknown impurity criterion')

    return round(impurity, 3)


'''a function to compute information gain after splitting based on a certain feature, 
this function is obtained from 'https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation/'''


def comp_feature_information_gain(df, target, descriptive_feature, split_criterion):
    """
    This function calculates information gain for splitting on
    a particular descriptive feature for a given dataset
    and a given impurity criteria.
    Supported split criterion: 'entropy', 'gini'
    """

    target_entropy = compute_impurity(df[target], split_criterion)

    # we define two lists below:
    # entropy_list to store the entropy of each partition
    # weight_list to store the relative number of observations in each partition
    entropy_list = list()
    weight_list = list()

    # loop over each level of the descriptive feature
    # to partition the dataset with respect to that level
    # and compute the entropy and the weight of the level's partition
    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_impurity(df_feature_level[target], split_criterion)
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(round(weight_level, 3))

    # compute either the gini split or the entropy split of a feature
    remaining_features_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    information_gain = target_entropy - remaining_features_impurity
    return information_gain


# a function to compute WOE and IV:
def calculate_woe_iv(ds, ffeatures, target):
    IV = []
    for feat in ffeatures:
        lst = []
        for i in range(ds[feat].nunique()):
            val = list(ds[feat].unique())[i]
            lst.append({
                'Value': val,
                'All': ds[ds[feat] == val].count()[feat],
                'Good': ds[(ds[feat] == val) & (ds[target] == 0)].count()[feat],
                'Bad': ds[(ds[feat] == val) & (ds[target] == 1)].count()[feat]
            })

        df = pd.DataFrame(lst)
        df['Distr_Good'] = df['Good'] / df['Good'].sum()
        df['Distr_Bad'] = df['Bad'] / df['Bad'].sum()
        df['WoE'] = np.log(df['Distr_Good'] / df['Distr_Bad'])
        df = df.replace({'WoE': {np.inf: 0, -np.inf: 0}})
        df['IV'] = (df['Distr_Good'] - df['Distr_Bad']) * df['WoE']  # for a single bin
        IV.append(df['IV'].sum())  # for the whole feature

    return IV


# python functions to compute reducts and core of large datasets based on feature selection
def compute_features_importance(out_dir, ds, X_train, y_train, ffeatures, target_name,
                                model_xgboost, model_logit, df_discretized, file_name):
    results_df = pd.DataFrame()
    mean_threshold = {}

    if ds in ['spect', 'Zoo']:
        selector_chi_anova = SelectKBest(chi2, k='all').fit(X_train, y_train)
    else:
        selector_chi_anova = SelectKBest(f_classif, k='all').fit(X_train, y_train)
    # Embedded selectors (for logit and xgboost):
    lasso_selector = SelectFromModel(estimator=model_logit, prefit=True, threshold=-np.inf, max_features=len(ffeatures))
    tree_selector = SelectFromModel(estimator=model_xgboost, prefit=True, threshold=-np.inf,
                                    max_features=len(ffeatures))

    # TuRF
    # make sure to have the 'pct' other than 0.5, if the #of features is odd
    trelief_selector = TuRF(core_algorithm="ReliefF", n_features_to_select=len(ffeatures),
                            pct=0.3, verbose=True).fit(X_train.values, y_train.values, ffeatures)

    # Information gain:
    info_gain_entropy = []
    info_gain_gini = []
    # split_criterion = 'entropy'
    for f in ffeatures:
        feature_info_gain_entropy = comp_feature_information_gain(df_discretized, target_name, f, 'entropy')
        feature_info_gain_gini = comp_feature_information_gain(df_discretized, target_name, f, 'gini')
        info_gain_entropy.append(feature_info_gain_entropy)
        info_gain_gini.append(feature_info_gain_gini)

    IV = calculate_woe_iv(df_discretized, ffeatures, target_name)

    if ds in ['spect', 'Zoo']:
        results_df['Chi2'] = selector_chi_anova.scores_
    else:
        results_df['ANOVA'] = selector_chi_anova.scores_

    results_df['embedded_logit'] = lasso_selector.estimator.coef_[0]
    results_df['embedded_xgboost'] = tree_selector.estimator.feature_importances_
    results_df['Information_gain_entropy'] = pd.Series(info_gain_entropy)
    results_df['Information_gain_gini'] = pd.Series(info_gain_gini)
    # calculate Information Value:
    results_df['IV'] = pd.Series(IV)
    results_df['TuRF'] = trelief_selector.feature_importances_
    
    #to check for negative scores and shift the column prior to values normalization
    l_negative = results_df.columns[(results_df < 0).any()].tolist()
    if l_negative:
        for x in l_negative:
          min = results_df[l_negative].min() 
          if min < 0:
            results_df[x] = results_df[x].apply(lambda x: x +abs(min))
    std_scaler = MinMaxScaler().fit(results_df)
    results_df = pd.DataFrame(std_scaler.transform(results_df), columns=results_df.columns)
    # get the mean score of each criteria
    results_df_mean = results_df[[x for x in results_df.columns.values]].mean()
    results_df = results_df.append(results_df_mean, ignore_index=True)
    results_df['feature'] = ffeatures + ['mean_threshold']
    # rearrange columns to bring the 'feature' column to the front
    cols = results_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    results_df = results_df[cols]

    # the threshold in the mean of means
    Threshold_min = results_df_mean.min()
    Threshold_max = results_df_mean.max()
    results_df.to_csv(os.path.join(out_dir, 'features_scores_%s.csv' % (file_name)), sep=';')
    with open(os.path.join(out_dir, 'features_scores_%s.csv' % (file_name)), 'a') as fout:
        fout.write('\n')
        fout.write('Threshold_min;%s\n' % (Threshold_min))
        fout.write('Threshold_max;%s\n' % (Threshold_max))


