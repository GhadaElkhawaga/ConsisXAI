#Import Libraries
import ast
import os
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.retrieval import retrieve_artefact, retrieve_vector, get_imp_features
from preprocessing.datasets_preprocessing import scale_update, compute_class_counts
from preprocessing.Indispensable_feats_gen import compute_features_importance
from modeling.Models_training import train_models
from evaluation.Consistency_measures import calculate_abic
from evaluation.Consistency_ratios import compute_reducts_core, ComputeRatio


experiments_dir = os.path.join(out_dir, 'experimental_measurements')
if not os.path.exists(experiments_dir):
  os.makedirs(experiments_dir)

for ds in datasets:   
  file_name = info_df.loc[info_df['Dataset_name']==ds, 'files_names'].values[0]
  target_name = info_df.loc[info_df['Dataset_name']==ds, 'targets_names'].values[0]   
  num_cols = info_df.loc[info_df['Dataset_name']==ds, 'NumericalCols'].values[0]
  if ds in ['TruckFailure','Spam','Climate','Diabetic','Ionosphere','spect']:
    cat_cols = []
  else:
    cat_cols = info_df.loc[info_df['Dataset_name']==ds, 'CategoricalCols'].values[0]
  
  
  df = retrieve_artefact(logs_dir,'preprocessed_not_discretized_dataset_%s' %(file_name), '.csv', ',')
  df_discretized = retrieve_artefact(logs_dir,'discretized_dataset_%s' %(file_name), '.csv', ',')
  model_logit = retrieve_artefact(logs_dir,'model_logit_%s' %(file_name),'.pickle')
  model_xgboost = retrieve_artefact(logs_dir,'model_xgboost_%s' %(file_name),'.pickle')
  model_rf = retrieve_artefact(logs_dir,'model_rf_%s' %(file_name),'.pickle')
  model_gbm = retrieve_artefact(logs_dir,'model_gbm_%s' %(file_name),'.pickle')
  y= df[target_name]
  X = df.drop([target_name], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ffeatures = X_train.columns.tolist()
  #scaler = MinMaxScaler()
  #X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=ffeatures)
  compute_features_importances(ds,X_train,y_train,ffeatures, target_name,model_xgboost,model_logit, model_rf, model_gbm, df_discretized, cat_cols, num_cols) 

  compute_reducts_core(file_name,ds)

  coreset = retrieve_vector(os.path.join(logs_dir,'core_%s_python.json' %(file_name)))
  
  if  not coreset:
    continue
  else:
    core_Feats_indices = coreset
    core_feats = [ffeatures[i] for i in coreset] 
  
  all_reds = retrieve_vector(os.path.join(logs_dir,'reds_%s_python.json' %(file_name)))
  lengths_list = [len(l) for l in all_reds.values() if len(l)!= len(ffeatures)]
  shortest_red_idx = lengths_list.index(min(lengths_list)) #the shortest reduct is selected
  selected_red = list(all_reds.values())[shortest_red_idx]
  reduct_feats_indices = selected_red
  reduct_feats = [ffeatures[item] for item in selected_red]


  selected_by_xai_reduct = {}
  selected_by_xai_core = {}
  for cls_method in ['xgboost', 'logit', 'rf', 'gbm']:
    for xai_type in ['shap', 'perm']:
      key = '%s_%s_%s' %(ds,cls_method,xai_type)
      #print(key)
      selected_by_xai_core[key] = get_imp_features(file_name,\
                                                   ffeatures, cls_method, len(core_Feats_indices),xai_type)
      selected_by_xai_reduct[key] = get_imp_features(file_name,\
                                                     ffeatures, cls_method, len(reduct_feats_indices),xai_type)
  
  print('len core: %s' %(len(core_Feats_indices)))
  print('len reduct: %s' %(len(reduct_feats_indices)))
  scale_update(selected_by_xai_core)
  scale_update(selected_by_xai_reduct) 

  
  '''calculate the AIC/BIC values for the core/reduct of the complete features set'''
  sample_size = X_train.shape[0] 
  feats_len = len(ffeatures)
  

  reduct_ratio_dict, core_ratio_dict, AIC_reduct_dict, AIC_core_dict, BIC_reduct_dict,\
  BIC_core_dict, interreduct_scores_sum, reduct_intersection,\
  intercore_scores_sum, core_intersection   = (defaultdict(dict) for i in range(10))
  
  reduct_ratio_exp_dict, core_ratio_exp_dict, AIC_reduct_exp_dict, AIC_core_exp_dict, \
    BIC_reduct_exp_dict, BIC_core_exp_dict, reduct_intersection, core_intersection = \
        (defaultdict(dict) for i in range(8))
        
  for key in selected_by_xai_reduct.keys():
    #calculations using intersection features, and their scores:
    reduct_ratio_dict[key] , interreduct_scores_sum[key], reduct_intersection[key] = ComputeRatio(selected_by_xai_reduct[key],reduct_feats,'regular')
    comp_intersect_reduct = len(reduct_feats_indices) - reduct_intersection[key]
    AIC_reduct_dict[key] = calculate_abic(sample_size, reduct_ratio_dict[key],comp_intersect_reduct,'AIC')#len(reduct_feats_indices)
    BIC_reduct_dict[key] = calculate_abic(sample_size, reduct_ratio_dict[key], comp_intersect_reduct,'BIC') #len(reduct_feats_indices) 
    core_ratio_dict[key], intercore_scores_sum[key], core_intersection[key] = ComputeRatio(selected_by_xai_core[key],core_feats,'regular')
    comp_intersect_core = len(core_feats) - core_intersection[key]
    AIC_core_dict[key] = calculate_abic(sample_size, core_ratio_dict[key], comp_intersect_core,'AIC') #len(core_Feats_indices)
    BIC_core_dict[key] = calculate_abic(sample_size, core_ratio_dict[key], comp_intersect_core,'BIC') #len(core_Feats_indices)
    
    # just for experimenting and plotting reasons (using intersection features and their count):
    reduct_ratio_exp_dict[key], reduct_intersection[key] = \
            ComputeRatio(selected_by_xai_reduct[key], reduct_feats,'experimental')
    AIC_reduct_exp_dict[key] = \
            calculate_abic(sample_size, reduct_ratio_exp_dict[key], comp_intersect_reduct, 'AIC')
    BIC_reduct_exp_dict[key] = \
            calculate_abic(sample_size, reduct_ratio_exp_dict[key], comp_intersect_reduct, 'BIC')
    core_ratio_exp_dict[key], core_intersection[key] = \
            ComputeRatio(selected_by_xai_core[key], core_feats, 'experimental')    
    AIC_core_exp_dict[key] = \
            calculate_abic(sample_size, core_ratio_exp_dict[key], comp_intersect_core, 'AIC')
    BIC_core_exp_dict[key] = \
            calculate_abic(sample_size, core_ratio_exp_dict[key], comp_intersect_core, 'BIC')

    
  
  measurements = ['reduct_ratio','core_ratio','AIC_reduct',\
                           'AIC_core','BIC_reduct','BIC_core',\
                           'intersect_reduct_scores_sum', '#feats_reduct_intersection',\
                           'intersect_core_scores_sum', '#feats_core_intersection']

  results_df = pd.DataFrame.from_dict([reduct_ratio_dict, core_ratio_dict, \
                          AIC_reduct_dict,AIC_core_dict,\
                          BIC_reduct_dict, BIC_core_dict,\
                          interreduct_scores_sum, reduct_intersection,\
                          intercore_scores_sum, core_intersection])
  results_df = pd.concat([pd.DataFrame(measurements), results_df], axis=1)
  results_df.columns.values[0] = 'Measurement'

  #consistency_df = pd.DataFrame()
  results_df.to_csv(os.path.join(logs_dir,'measurements_%s.csv' %(file_name)), sep=';', index=False)
  #with open(os.path.join(Folder, 'measurements_%s.csv' %(file_name)), 'a') as fout:
  with open(os.path.join(logs_dir,'measurements_%s.csv' %(file_name)), 'a') as fout:
        fout.write('\n')
        fout.write('%s;%s\n' % ('Reduct_feats_%s' %(file_name), reduct_feats))
        fout.write('%s;%s\n' % ('Core_feats_%s' %(file_name), core_feats))
        fout.write('\n')

  selected_feats_reduct = pd.DataFrame.from_dict([selected_by_xai_reduct])
  selected_feats_core = pd.DataFrame.from_dict([selected_by_xai_core])
  selected_feats_df = pd.concat([selected_feats_reduct, selected_feats_core], axis=0)
  selected_feats_df.reset_index(inplace=True, drop=True)
  idx = pd.DataFrame(['selected_by_xai_reduct','selected_by_xai_core'])

  selected_feats_df = pd.concat([idx, selected_feats_df], axis=1)
  selected_feats_df.to_csv(os.path.join(logs_dir,'measurements_%s.csv' %(file_name)), sep=';', mode='a', index=False)
  
 
  print('AIC_reduct %s' %(AIC_reduct_dict[key]))  
  print('core_ratio_dict %s' %(core_ratio_dict[key]))
  print('AIC_core %s' %(AIC_core_dict[key]))
  
  
  measurements_exp = ['reduct_ratio', 'core_ratio', 'AIC_reduct',
                    'AIC_core', 'BIC_reduct', 'BIC_core',
                    '#feats_reduct_intersection', '#feats_core_intersection']

  results_df_exp = pd.DataFrame.from_dict([reduct_ratio_exp_dict, core_ratio_exp_dict,
                                         AIC_reduct_exp_dict, AIC_core_exp_dict,
                                         BIC_reduct_exp_dict, BIC_core_exp_dict,
                                         reduct_intersection, core_intersection])
  results_df_exp = pd.concat([pd.DataFrame(measurements), results_df], axis=1)
  results_df_exp.columns.values[0] = 'Measurement'

  results_df_exp.to_csv(os.path.join(experiments_dir, 'measurements_experimental_%s.csv' %(file_name)),
                      sep=';', index=False)
  with open(os.path.join(experiments_dir, 'measurements_experimental_%s.csv' % (file_name)), 'a') as fout:
        fout.write('\n')
        fout.write('%s;%s\n' % ('Reduct_feats_%s' % (file_name), reduct_feats))
        fout.write('%s;%s\n' % ('Core_feats_%s' % (file_name), core_feats))
        fout.write('\n')

  
  selected_feats_df.to_csv(os.path.join(experiments_dir, 'measurements_experimental_%s.csv' %(file_name)),
                             sep=';', mode='a', index=False)



