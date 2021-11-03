import pickle
import pandas as pd
from scipy.io import arff
import json
import os
import numpy as np


def retrieve_vector(folder, fname):
    with open(os.path.join(folder,fname)) as f:
        return json.load(f)


# a function to retrieve artefacts
def retrieve_artefact(folder, filename, file_end, vals_sep=None):
    retrieved_file = filename + file_end
    file = os.path.join(folder, retrieved_file)
    if '.pickle' in file_end:
        with open(file, 'rb') as fin:
            retrieved_artefact = pickle.load(fin)
    elif '.data' in file_end:
        retrieved_artefact = pd.read_csv(file, sep=vals_sep, engine='python')
    elif '.arff' in file_end:
        data = arff.loadarff(file)
        retrieved_artefact = pd.DataFrame(data[0])
    else:
        retrieved_artefact = pd.read_csv(file, sep=vals_sep, encoding='ISO-8859-1', engine='python')
    return retrieved_artefact


# a function to get important features according to each XAI method:
def get_imp_features(folder, file_name, ffeatures, cls_method, num_retrieved_feats, xai_type=None):
    frmt_str = {
        'ALE': 'ALE_pred_explainer_%s_%s' % (file_name, cls_method),
        'shap': 'shap_values_%s_%s' % (file_name, cls_method),
        'perm': 'permutation_importance_%s_%s_training' % (file_name, cls_method)}
    # num_retrieved_feats = threshold * len(ffeatures.iloc[:,0])
    # number of features to be retrieved from the imp features set
    if xai_type != None:
        feats_df = retrieve_artefact(folder, frmt_str[xai_type], '.csv', ';')
        local_feats = num_retrieved_feats
        if xai_type == 'ALE':
            feats_df_sorted = pd.DataFrame(feats_df, columns=feats_df.columns)
            imp_feats_scores = feats_df_sorted.iloc[:local_feats, -1]
            # adjust local_feats if the number of features passed to ALE is different than the whole feature set
            local_feats = min(num_retrieved_feats, len(feats_df_sorted.iloc[:, 0]))
        elif xai_type == 'shap':
            avg_shap_values = pd.DataFrame(np.mean(np.abs(feats_df.values), 0), columns=['shap_vals'])
            feats_df = pd.concat([pd.Series(ffeatures), avg_shap_values], axis=1)
            feats_df_sorted = feats_df.sort_values(by='shap_vals', ascending=False)
        elif xai_type == 'perm':
            feats_df_sorted = pd.DataFrame(data=feats_df.values, columns=feats_df.columns)
        else:
            raise ValueError('XAI method not found')

    imp_feats_names = feats_df_sorted.iloc[:local_feats, 0]
    if xai_type in ['shap', 'perm']:
        imp_feats_scores = feats_df_sorted.iloc[:local_feats, 1]
    imp_feats = dict(zip(imp_feats_names, imp_feats_scores))
    return imp_feats




