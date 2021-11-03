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



datasets_folder = 'datasets_files'
if not os.path.exists(os.path.join(datasets_folder)):
  os.makedirs(os.path.join(datasets_folder))
params_dir = 'optimal_params'
if not os.path.exists(os.path.join(params_dir)):
  os.makedirs(os.path.join(params_dir))
models_folder = 'models'
if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), models_folder)):
  os.makedirs(os.path.join(os.path.dirname(os.getcwd()), models_folder))
out_dir = 'output_files'
if not os.path.exists(os.path.join(out_dir)):
  os.makedirs(os.path.join(out_dir))
reds_core_dir = os.path.join(out_dir, 'reducts_core')
if not os.path.exists(reds_core_dir):
  os.makedirs(reds_core_dir)
measurements_dir = os.path.join(out_dir, 'measurements')
if not os.path.exists(measurements_dir):
  os.makedirs(measurements_dir)
permutation_dir = 'permutation_outputs'
if not os.path.exists(os.path.join(permutation_dir)):
        os.makedirs(os.path.join(permutation_dir))
shap_dir = 'shap_outputs'
if not os.path.exists(os.path.join(shap_dir)):
  os.makedirs(os.path.join(shap_dir))
ale_dir = 'ale_outputs'
if not os.path.exists(os.path.join(ale_dir)):
  os.makedirs(os.path.join(ale_dir))

info_df = retrieve_artefact(datasets_folder, 'datasets_info', '.csv', ',')
datasets = ['BreastC', 'Wine', 'Zoo', 'Diabetic', 'Ionosphere', 'spect', 'Vehicle']
out_folder = {'ALE': 'ale_outputs', 'shap': 'shap_outputs', 'perm': 'permutation_outputs'}

compute_class_counts(datasets_folder, datasets, info_df, out_dir)
train_models(models_folder, datasets, info_df, params_dir, datasets_folder, out_dir)

for ds in datasets:
    file_name = info_df.loc[info_df['Dataset_name'] == ds, 'files_names'].values[0]
    target_name = info_df.loc[info_df['Dataset_name'] == ds, 'targets_names'].values[0]
    file_name = info_df.loc[info_df['Dataset_name'] == ds, 'files_names'].values[0]
    ffeatures = ast.literal_eval(info_df.loc[info_df['Dataset_name'] == ds, 'Columns'].values[0])
    training_size = ast.literal_eval(info_df.loc[info_df['Dataset_name'] == ds, 'Training_size'].values[0])
    testing_size = ast.literal_eval(info_df.loc[info_df['Dataset_name'] == ds, 'Testing_size'].values[0])
    print(ds)
    df = retrieve_artefact(datasets_folder, 'preprocessed_not_discretized_dataset_%s' % (file_name), '.csv', ',')
    df_discretized = retrieve_artefact(datasets_folder, 'discretized_dataset_%s' % (file_name), '.csv', ',')
    model_logit = retrieve_artefact(models_folder, 'model_logit_%s' % (file_name), '.pickle')
    model_xgboost = retrieve_artefact(models_folder, 'model_xgboost_%s' % (file_name), '.pickle')
    unfitted_model_logit = retrieve_artefact(models_folder, 'unfitted_model_logit_%s' % (file_name), '.pickle')
    unfitted_model_xgboost = retrieve_artefact(models_folder, 'unfitted_model_xgboost_%s' % (file_name), '.pickle')
    y = df[target_name]
    X = df.drop([target_name], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ffeatures = ast.literal_eval(info_df.loc[info_df['Dataset_name'] == ds, 'Columns'].values[0])

    if ds == 'Scene':
        ffeatures = [f for f in ffeatures if f not in ["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"]]
    else:
        ffeatures.remove(target_name)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=ffeatures)

    compute_features_importance(out_dir, ds, X_train, y_train, ffeatures, target_name, model_xgboost, model_logit,
                                 df_discretized, file_name)
    compute_reducts_core(out_dir, file_name, reds_core_dir)

    coreset = retrieve_vector(reds_core_dir, 'core_%s_python.json' %(file_name))
    if not coreset:
        continue
        # raise ValueError('a core can not be represented by an empty set')
    else:
        core_Feats_indices = coreset
        core_feats = [ffeatures[i] for i in coreset]
    all_reds = retrieve_vector(reds_core_dir, 'reds_%s_python.json' % (file_name))
    lengths_list = [len(l) for l in all_reds.values() if len(l) != len(ffeatures)]
    shortest_red_idx = lengths_list.index(min(lengths_list))  # the shortest reduct is selected
    selected_red = list(all_reds.values())[shortest_red_idx]
    reduct_feats_indices = selected_red
    reduct_feats = [ffeatures[item] for item in selected_red]
    selected_by_xai_reduct = {}
    selected_by_xai_core = {}

    for cls_method in ['xgboost', 'logit']:
        for xai_type in ['shap', 'perm', 'ALE']:
            key = '%s_%s_%s' % (ds, cls_method, xai_type)
            # print(key)
            selected_by_xai_core[key] = get_imp_features(out_folder[xai_type], file_name,
                                                         ffeatures, cls_method,
                                                         len(core_Feats_indices), xai_type)
            selected_by_xai_reduct[key] = get_imp_features(out_folder[xai_type], file_name,
                                                           ffeatures, cls_method,
                                                           len(reduct_feats_indices), xai_type)
    for x in [selected_by_xai_core, selected_by_xai_reduct]:
        scale_update(x)
    #calculate the AIC/BIC values for the core/reduct of the complete features set
    feats_len = len(ffeatures)
    sample_size = training_size[0]
    reduct_ratio_dict, core_ratio_dict, AIC_reduct_dict, AIC_core_dict, \
    BIC_reduct_dict, BIC_core_dict, interreduct_scores_sum, reduct_intersection, \
    intercore_scores_sum, core_intersection = (defaultdict(dict) for i in range(10))
    for key in selected_by_xai_reduct.keys():
        # calculations using intersection features, and their scores:
        reduct_ratio_dict[key], interreduct_scores_sum[key], reduct_intersection[key] =\
          ComputeRatio(selected_by_xai_reduct[key], reduct_feats, 'regular')
        comp_intersect_reduct = len(reduct_feats_indices) - reduct_intersection[key]
        AIC_reduct_dict[key] = calculate_abic(sample_size, reduct_ratio_dict[key],
                                              comp_intersect_reduct, 'AIC')
        BIC_reduct_dict[key] = calculate_abic(sample_size, reduct_ratio_dict[key],
                                              comp_intersect_reduct, 'BIC')
        core_ratio_dict[key], intercore_scores_sum[key], core_intersection[key] =\
          ComputeRatio(selected_by_xai_core[key], core_feats, 'regular')
        comp_intersect_core = len(core_feats) - core_intersection[key]
        AIC_core_dict[key] = calculate_abic(sample_size, core_ratio_dict[key],
                                            comp_intersect_core, 'AIC')
        BIC_core_dict[key] = calculate_abic(sample_size, core_ratio_dict[key],
                                            comp_intersect_core, 'BIC')

    measurements = ['reduct_ratio', 'core_ratio', 'AIC_reduct',
                    'AIC_core', 'BIC_reduct', 'BIC_core',
                    'intersect_reduct_scores_sum', '#feats_reduct_intersection',
                    'intersect_core_scores_sum', '#feats_core_intersection']
    results_df = pd.DataFrame.from_dict([reduct_ratio_dict, core_ratio_dict,
                                         AIC_reduct_dict, AIC_core_dict,
                                         BIC_reduct_dict, BIC_core_dict,
                                         interreduct_scores_sum, reduct_intersection,
                                         intercore_scores_sum, core_intersection])
    results_df = pd.concat([pd.DataFrame(measurements), results_df], axis=1)
    results_df.columns.values[0] = 'Measurement'  
    results_df.to_csv(os.path.join(measurements_dir, 'measurements_%s.csv' %(file_name)),
                      sep=';', index=False)
    with open(os.path.join(measurements_dir, 'measurements_%s.csv' %(file_name)), 'a') as fout:
        fout.write('\n')
        fout.write('%s;%s\n' % ('Reduct_feats_%s' %(file_name), reduct_feats))
        fout.write('%s;%s\n' % ('Core_feats_%s' %(file_name), core_feats))
        fout.write('\n')
    selected_feats_reduct = pd.DataFrame.from_dict([selected_by_xai_reduct])
    selected_feats_core = pd.DataFrame.from_dict([selected_by_xai_core])
    selected_feats_df = pd.concat([selected_feats_reduct, selected_feats_core], axis=0)
    selected_feats_df.reset_index(inplace=True, drop=True)
    idx = pd.DataFrame(['selected_by_xai_reduct', 'selected_by_xai_core'])
    selected_feats_df = pd.concat([idx, selected_feats_df], axis=1)
    selected_feats_df.to_csv(os.path.join(measurements_dir, 'measurements_%s.csv' %(file_name)),
                             sep=';', mode='a', index=False)


