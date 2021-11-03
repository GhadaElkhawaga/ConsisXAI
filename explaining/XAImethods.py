from sklearn.inspection import permutation_importance
import shap
import numpy as np
from alibi.explainers import ALE
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from explaining.ALEcomputations import feats_impurity


def explain_predictions(xai_method, file_name, clf, ffeatures, X, y, clf_type):
  if xai_method == 'perm':
    Permutation_importance_analysis('permutation_outputs',
                                    file_name, clf, ffeatures, X, y, clf_type)
  elif xai_method == 'shap':
    shap_global('shap_outputs', file_name, clf, X, ffeatures, clf_type)
  else:
    ALE_Computing('ale_outputs', file_name, ffeatures, X, y, clf, clf_type)


def Permutation_importance_analysis(permutation_dir, file_name, cls, ffeatures, x, y, cls_method):
    permutation_file_name = 'permutation_importance_%s_%s' % (file_name, cls_method)
    training_result = permutation_importance(cls, x, y, n_repeats=10, random_state=42, n_jobs=-1)

    cols = ['Feature', 'importances(mean)', 'importances(std)', 'importances']
    df_res_train = pd.DataFrame(zip(ffeatures, training_result.importances_mean, \
                                    training_result.importances_std, training_result.importances), \
                                columns=cols)
    df_train_sorted = df_res_train.sort_values('importances(mean)', ascending=False)
    df_train_sorted.to_csv(os.path.join(permutation_dir, '%s_training.csv' % (permutation_file_name)), sep=';',
                           index=False)

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.boxplot(df_train_sorted.iloc[:3, 3].T, vert=False, labels=df_train_sorted.iloc[:3, 0])
    ax.set_title("Permutation Importances (train set)")
    plt.savefig(os.path.join(permutation_dir, '%s_training.png' % (permutation_file_name)), dpi=300,
                bbox_inches='tight');
    plt.close()

    plt.figure(figsize=(12, 8))
    rects = ax.barh(np.arange(0, len(df_train_sorted['Feature'])), df_train_sorted.iloc[:, 1], align='center',
                    alpha=0.5)
    plt.yticks(np.arange(0, len(df_train_sorted['Feature'])), df_train_sorted.iloc[:, 0])
    plt.xlabel('Importances')
    plt.title("Permutation Importances (train set)");
    plt.savefig(os.path.join(permutation_dir,'%s_training2.png' % (permutation_file_name)), dpi=300,
                bbox_inches='tight');
    plt.close()


def shap_global(shap_dir, file_name, cls, X, ffeatures, cls_method):
    if shap.__version__ >= str(0.37):
        explainer = shap.Explainer(cls, X, feature_names=ffeatures)
    else:
        if cls_method == 'xgboost':
            explainer = shap.TreeExplainer(cls)
            print('shap explainer model_output: {0}'.format(explainer.model_output))
            print('shap explainer feature_perturbation: {0}'.format(explainer.feature_perturbation))
        else:
            explainer = shap.LinearExplainer(cls, X)

    if cls_method == 'xgboost':
        shap_values = explainer.shap_values(X, check_additivity=False)
    else:
        shap_values = explainer.shap_values(X)

    print('type of shap values: {0}'.format(type(shap_values)))

    out1 = os.path.join(shap_dir,
        'shap_explainer_%s_%s.pickle' % (file_name, cls_method))
    with open(out1, 'wb') as output:
        pickle.dump(explainer, output)

    shap_csv = os.path.join(shap_dir, 'shap_values_%s_%s.csv' % (file_name, cls_method))
    pd.DataFrame(shap_values).to_csv(shap_csv, sep=';', index=False)

    shap_data = os.path.join(shap_dir,
        'shap_values_%s_%s.pickle' % (file_name, cls_method))
    with open(shap_data, 'wb') as fout:
        pickle.dump(shap_values, fout)

    print('summary plot of shap values - normal')
    # shap.initjs()
    if shap.__version__ >= str(0.37):
        shap.plots.beeswarm(shap_values, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, max_display=10, show=False)
    plt.savefig(os.path.join(shap_dir, 'Shap values_normal_%s_%s.png' % (file_name, cls_method)),
                dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    print('-' * 100)

    print('summary plot of shap values - bar')
    # shap.initjs()
    if shap.__version__ >= str(0.37):
        shap.plots.bar(shap_values, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, plot_type='bar', show=False, max_display=10)

    plt.savefig(
        os.path.join(shap_dir, 'Shap values_bar_%s_%s.png' % (file_name, cls_method)),
        dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()

    del explainer
    del shap_values


def ALE_Computing(ale_dir, file_name, ffeatures, x, y, cls, cls_method, target_names=None):
    ale_pred = ALE(cls.predict_proba, feature_names=ffeatures)
    print('done with the explainer initiation')

    explainer_pred = ale_pred.explain(x)
    print('explanations generated')

    explainer_pred_data = os.path.join(ale_dir, 'ALE_pred_explainer_%s_%s.pickle' % (file_name, cls_method))
    with open(explainer_pred_data, 'wb') as output:
        pickle.dump(explainer_pred, output)

    # saving values at which ale values are computed for each feature in csv files:
    cols = ['Feature', 'ALE_vals', 'feature_values_for_ALECalc', 'ale0', 'entropy']
    ent = feats_impurity(explainer_pred.feature_names, explainer_pred.ale_values)
    df_res_exp_pred = pd.DataFrame(
        zip(explainer_pred.feature_names, explainer_pred.ale_values, explainer_pred.feature_values,
            explainer_pred.ale0, pd.Series(ent)), \
        columns=cols)

    df_res_exp_pred_file = os.path.join(ale_dir, 'ALE_pred_explainer_%s_%s.csv' % (file_name, cls_method))
    df_res_exp_pred_sorted = df_res_exp_pred.sort_values('entropy')
    df_res_exp_pred_sorted.to_csv(df_res_exp_pred_file, sep=';', index=False)

    plt.figure(figsize=(12, 8))
    plt.barh(np.arange(0, len(df_res_exp_pred_sorted['entropy'])), df_res_exp_pred_sorted.iloc[:, 4], align='center',
             alpha=0.5)
    plt.yticks(np.arange(0, len(df_res_exp_pred_sorted['entropy'])), df_res_exp_pred_sorted.iloc[:, 0])
    plt.xlabel('impurity')
    plt.title("quality of split");
    plt.savefig(os.path.join(ale_dir, '%s_impuritybasedonALE.png' % (file_name)), dpi=300,
                bbox_inches='tight');
    plt.close()

