from sklearn.inspection import permutation_importance
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle



def shap_global(file_name,cls, X, ffeatures,cls_method):
    
    explainer = shap.Explainer(cls, X)
    shap_vals = explainer(X)
    shap_values = explainer.shap_values(X)
          
    print('\n type of shap values: {0}'.format(type(shap_values)))


    out1 = os.path.join(logs_dir, 'shap_explainer_%s_%s.pickle' %(file_name,cls_method))
    with open(out1, 'wb') as output:
        pickle.dump(explainer, output)

    shap_csv = os.path.join(logs_dir, 'shap_values_%s_%s.csv' %(file_name,cls_method) )
    shap_df = pd.DataFrame(shap_values, columns =ffeatures)
    shap_df.to_csv(shap_csv, sep=';', index=False)

    shap_data = os.path.join(logs_dir, 'shap_values_%s_%s.pickle' %(file_name,cls_method))
    with open(shap_data, 'wb') as fout:
        pickle.dump(shap_values, fout)
 

    print('summary plot of shap values - normal')
    # shap.initjs()
    if shap.__version__ >= str(0.37):
        shap.plots.beeswarm(shap_vals, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, max_display=10, show=False)
    plt.savefig(os.path.join(logs_dir, 'Shap values_normal_%s_%s.png' %(file_name,cls_method)),
                dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    print('-' * 100)

    print('summary plot of shap values - bar')
    # shap.initjs()
    if shap.__version__ >= str(0.37):
        shap.plots.bar(shap_vals, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, plot_type='bar', show=False, max_display=10)

    plt.savefig(
        os.path.join(logs_dir, 'Shap values_bar_%s_%s.png' %(file_name,cls_method)),
        dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()

    del explainer
    del shap_values

##########################################################################################################################################
def Permutation_importance_analysis(file_name,cls, ffeatures, x, y, cls_method):
       
    permutation_file_name = 'permutation_importance_%s_%s' %(file_name,cls_method)
    training_result = permutation_importance(cls,x, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_sorted =  training_result.importances_mean.argsort()


    cols = ['Feature', 'importances(mean)', 'importances(std)', 'importances']
    df_res_train = pd.DataFrame(zip(ffeatures, training_result.importances_mean, \
                                    training_result.importances_std, training_result.importances), \
                                columns=cols)
    df_train_sorted = df_res_train.sort_values('importances(mean)', ascending=False)
    df_train_sorted.to_csv(os.path.join(logs_dir,'%s_training.csv' % (permutation_file_name)), sep=';', index=False)

    plt.figure(figsize=(18,8))
    plt.barh(np.arange(0, len(df_train_sorted['Feature'])), df_train_sorted.iloc[:, 1], align='center', alpha=0.5)
    plt.yticks(np.arange(0, len(df_train_sorted['Feature'])), df_train_sorted.iloc[:, 0])
    plt.xlabel('Importances')
    plt.title("Permutation Importances (train set)");
    plt.savefig(os.path.join(logs_dir,'%s_training2.png' % (permutation_file_name)), dpi=300,
                bbox_inches='tight');
    plt.close()

###############################################################################################################################################
def explain_predictions(xai_method, file_name, clf, ffeatures, X, y, clf_type):
  if xai_method == 'perm':
    Permutation_importance_analysis(file_name,clf, ffeatures, X, y, clf_type)
  elif xai_method == 'shap':
    shap_global(file_name,clf, X, ffeatures,clf_type)
