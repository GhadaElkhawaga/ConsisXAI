import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import SelectFromModel, SelectKBest, SequentialFeatureSelector, RFECV, f_classif, RFE, chi2
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix,precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour
from skrebate import TuRF


def refine_probabilities(l):
  vals = []
  for x in range(len(l)):
        v = l[x]
        if v < 0:
          v = 1 - l[x] 
        vals.append(v)
  return vals

##################################################################################################################################################
def get_counts(l1,l2):
  count_c0, count_c1 = 0, 0
  for x in range(len(l1)):
    if l1[x] > l2[x]:
      count_c0 += 1
    elif l1[x] == l2[x]:
      selected = random.randint(0,1)
      if selected == 0:
        count_c0 += 1
      else: 
        count_c1 += 1
    else:
      count_c1 += 1    
  return count_c0, count_c1

#############################################################################################################################################
def compute_entropy(l):
  return -1 * (sum([v*log(v,2) for v in l if v!=0]))

##################################################################################################################################################
def feats_impurity(feature_names, ale_values):
  ent = []
  for feat in range(len(feature_names)):
      vals_c0, vals_c1, probs  = ([] for i in range(3))
      ent_feat = 0
      cat_count = len(ale_values[feat][:,0])      
      vals_c0 = refine_probabilities(ale_values[feat][:,0]) 
      vals_c1 = refine_probabilities(ale_values[feat][:,1])
      #count_c0, count_c1 = get_counts(vals_c0, vals_c1)
      for x in get_counts(vals_c0, vals_c1):
        probs.append(x/cat_count)
      for y in probs:
        if y == 0:
          continue
        else:
          ent_feat += (y*(log(y,2))) 
      ent_feat = ent_feat * -1
      #check correctness of computations:
      if ent_feat > log(cat_count,2):
        print('Maybe there is an error in calculations')
      ent.append(ent_feat)
  return ent
  
##############################################################################################################################################  
'''a function to compute information gain after splitting based on a certain feature, 
this function is obtained from 'https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation/'''

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
    return(round(impurity, 3))

#####################################################################################################################################
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
    #compute either the gini split or the entropy split of a feature
    remaining_features_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))   
    information_gain = target_entropy - remaining_features_impurity
    return(information_gain)
    
######################################################################################################################################## #   
#a function to compute WOE and IV:
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
    df['IV'] = (df['Distr_Good'] - df['Distr_Bad']) * df['WoE'] # for a single bin
    IV.append(df['IV'].sum()) # for the whole feature   
  return IV
  
################################################################################################################################################  
def chi_anova(ds, X_train, y_train, catCols, numCols):
    numCols = numCols
    numerical_feats_df = X_train[numCols]
    selector_anova = SelectKBest(f_classif, k='all').fit(numerical_feats_df, y_train)
    scores_feats_anova = pd.Series(data=selector_anova.scores_, index=numCols)
    if catCols:
      catCols = catCols
      categorical_feats_df = X_train[catCols]
      selector_chi = SelectKBest(chi2, k='all').fit(categorical_feats_df, y_train)
      scores_feats_chi = pd.Series(data=selector_chi.scores_, index=catCols)
      scores_feats_chi_anova = pd.concat([scores_feats_anova, scores_feats_chi], names=["Chi/ANOVA"]).to_frame()
      scores_feats_chi_anova['feature'] = scores_feats_chi_anova.index  
    else:
        scores_feats_chi_anova = pd.DataFrame({'feature': X_train.columns.tolist(), "Chi/ANOVA": scores_feats_anova})     
    return  scores_feats_chi_anova
    
###############################################################################################################################################
#python functions to compute reducts and core of large datasets based on feature selection
def compute_features_importances(ds, X_train, y_train, ffeatures,\
                                 target_name,model_xgboost,model_logit,model_rf, \
                                 model_gbm, df_discretized, catCols, numCols):
      results_df = pd.DataFrame()
      mean_threshold = {}

      #get chi scores combined with anova scores for categorical and numerical features, respectively.
      scores_feats_chi_anova = chi_anova(ds, X_train, y_train, catCols, numCols)
      #Embedded selectors (for logit and xgboost):
      lasso_selector = SelectFromModel(estimator=model_logit, prefit=True, threshold=-np.inf, max_features=len(ffeatures))
      tree_selector = SelectFromModel(estimator=model_xgboost, prefit=True, threshold=-np.inf, max_features=len(ffeatures))
      rf_selector = SelectFromModel(estimator=model_rf, prefit=True, threshold=-np.inf, max_features=len(ffeatures))
      gbm_selector = SelectFromModel(estimator=model_gbm, prefit=True, threshold=-np.inf, max_features=len(ffeatures))
      
      #Information gain:
      info_gain_entropy = []
      info_gain_gini = []
      #split_criterion = 'entropy'
      for f in ffeatures:
        feature_info_gain_entropy = comp_feature_information_gain(df_discretized, target_name, f, 'entropy')
        feature_info_gain_gini = comp_feature_information_gain(df_discretized, target_name, f, 'gini')
        info_gain_entropy.append(feature_info_gain_entropy)
        info_gain_gini.append(feature_info_gain_gini)      
      IV = calculate_woe_iv(df_discretized, ffeatures, target_name)
      
      results_df['embedded_logit'] = lasso_selector.estimator.coef_[0]
      results_df['embedded_xgboost'] = tree_selector.estimator.feature_importances_
      results_df['embedded_rf'] = rf_selector.estimator.feature_importances_
      results_df['embedded_gbm'] = gbm_selector.estimator.feature_importances_
      results_df['Information_gain_entropy'] = pd.Series(info_gain_entropy)
      results_df['Information_gain_gini'] = pd.Series(info_gain_gini)
      #calculate Information Value:
      results_df['IV'] = pd.Series(IV)
      results_df['feature'] = ffeatures
      results_df = results_df.merge(scores_feats_chi_anova, how='left', on='feature')

      try:
        #TuRF 
        trelief_selector = TuRF(core_algorithm="ReliefF",n_features_to_select=len(ffeatures),\
                                pct=0.3,verbose=True).fit(X_train.values, y_train.values, ffeatures)          
        results_df['TuRF'] = trelief_selector.feature_importances_      
        df_cols = ['feature', 'TuRF', 'IV', 'Information_gain_gini',
                        'Information_gain_entropy', 'embedded_logit', 'embedded_xgboost','embedded_rf','embedded_gbm']
      except:
        df_cols = ['feature', 'IV', 'Information_gain_gini',
                        'Information_gain_entropy', 'embedded_logit', 'embedded_xgboost','embedded_rf','embedded_gbm']

      for x in results_df.columns:
          if x not in df_cols:
              results_df.rename(columns={x: 'Chi/ANOVA'}, inplace=True)
              break

      calc_cols = results_df.columns.tolist()
      calc_cols.remove('feature')
      results_df.drop('feature', 1, inplace=True)
      # shift columns containing negative values before normalization:
      l_negative = results_df.columns[(results_df < 0).any()].tolist()
      if l_negative:
          for x in l_negative:
              min = results_df[x].min()
              if min < 0:
                  results_df[x] = results_df[x].apply(lambda x: x + abs(min))

      #std_scaler = StandardScaler().fit(results_df)
      #std_scaler = MinMaxScaler().fit(results_df)
      for col in results_df.columns.tolist():
        results_df[col] = MinMaxScaler().fit_transform(results_df[col].values.reshape(-1,1))
      #results_df = pd.DataFrame(std_scaler.transform(results_df), columns=results_df.columns)
      #get the mean score of each criteria
      results_df_mean = results_df[[x for x in results_df.columns.values]].mean()
      results_df = results_df.append(results_df_mean, ignore_index=True)
      results_df['feature'] = ffeatures + ['mean_threshold']
      #rearrange columns to bring the 'feature' column to the front
      cols = results_df.columns.tolist()
      cols = cols[-1:] + cols[:-1]
      results_df = results_df[cols]
      
      #the threshold in the mean of means
      Threshold_min = results_df_mean.min()
      Threshold_max = results_df_mean.max()   
      results_df.to_csv(os.path.join(logs_dir, 'features_scores_%s.csv' %(file_name)), sep=';')
      with open(os.path.join(logs_dir, 'features_scores_%s.csv' %(file_name)), 'a') as fout:
          fout.write('\n')
          fout.write('Threshold_min;%s\n' % (Threshold_min))
          fout.write('Threshold_max;%s\n' % (Threshold_max))
         
    
