import numpy as np
import ast
import json
import os
from functools import reduce
from utils.retrieval import retrieve_artefact


def get_threshold(threshold_min, threshold_max):
    threshold_min_100 = int(threshold_min * 100)
    diff = threshold_max - threshold_min
    diff_100 = int(diff * 100)
    rest_diff = diff - (diff_100 / 100.0)
    l = []
    for i in range(threshold_min_100, diff_100):
        x = (i / 100.0) + rest_diff
        l.append(x)
    idx = np.random.randint(0, len(l))
    threshold = abs(threshold_min - l[idx])
    return threshold, len(l)


def compute_reds(params_df, threshold):
    reds = {}
    reds_weights = {}
    sorting_criteria = params_df.columns[2:]
    for sc in sorting_criteria:
        selected_feats = params_df.iloc[
            params_df.index[params_df[sc] >= threshold], params_df.columns.get_loc("feature")]
        selected_weights = params_df.iloc[params_df.index[params_df[sc] >= threshold],
                                          params_df.columns.get_loc(sc)]
        reds[sc] = selected_feats.index.values.tolist()
        reds_weights[sc] = selected_weights.tolist()
    return reds, reds_weights


def compute_core(reds):
    return list(reduce(set.intersection, (set(val) for val in reds.values())))


def compute_reducts_core(out_dir, file_name, reds_core_dir):
    params_df = retrieve_artefact(out_dir, 'features_scores_%s' %(file_name), '.csv', ';')
    threshold_min = ast.literal_eval(params_df.iloc[-2, 1])
    threshold_max = ast.literal_eval(params_df.iloc[-1, 1])
    params_df = params_df.drop(params_df.index[params_df['feature'] == 'mean_threshold'])
    threshold = threshold_min
    reds, reds_weights = compute_reds(params_df, threshold)
    core = compute_core(reds)
    i, length = 0, 1
    while (len(core) == 0) and (i < length):
        threshold, length = get_threshold(threshold_min, threshold_max)
        reds, reds_weights = compute_reds(params_df, threshold)
        core = compute_core(reds)
        i += 1
    # to calculate the weight of each feature in the core
    # taking the average of all weights of the same feature in different reducts
    core_feat_weights = []
    for x in core:
        red_feat_weights = []
        for key, val in reds.items():
            idx = val.index(x)
            red_feat_weights.append(reds_weights[key][idx])
        core_feat_weights.append(sum(red_feat_weights) / len(red_feat_weights))
    with open(os.path.join(reds_core_dir, 'reds_%s_python.json' %(file_name)), 'w') as outfile:
        json.dump(reds, outfile)
    with open(os.path.join(reds_core_dir, 'core_%s_python.json' %(file_name)), 'w') as outfile:
        json.dump(core, outfile)


def ComputeRatio(selectedsetByXYZ, originalset, flag=None):
    selected_feats = selectedsetByXYZ.keys()
    sum_numenator_set = 0
    original_set_sum = len(originalset) #each item in the reductset is assigned '1'
    numenator_set = set(originalset).intersection(set(selected_feats))
    intesection_vol = len(numenator_set)
    for x in numenator_set:
        sum_numenator_set += selectedsetByXYZ[x]
    if flag == 'regular':
        return sum_numenator_set/original_set_sum , sum_numenator_set, intesection_vol
    elif flag == 'experimental':
        return intesection_vol/original_set_sum, intesection_vol
    else:
        return
