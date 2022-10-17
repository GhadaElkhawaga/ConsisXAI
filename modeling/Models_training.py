import pickle
import ast
import os
from utils.retrieval import retrieve_artefact
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from explaining.XAImethods import explain_predictions
from evaluation.Consistency_measures import calculate_abic


def train_models(models_folder, datasets, info_df, params_dir, datasets_folder, out_dir):
    random_state = 42
    for ds in datasets:
        file_name = info_df.loc[info_df['Dataset_name'] == ds, 'files_names'].values[0]
        ffeatures = ast.literal_eval(info_df.loc[info_df['Dataset_name'] == ds, 'Columns'].values[0])
        target_name = info_df.loc[info_df['Dataset_name'] == ds, 'targets_names'].values[0]
        args = {}
        print(ds)
        df = retrieve_artefact(datasets_folder, 'preprocessed_not_discretized_dataset_%s' % (file_name), '.csv', ',')
        y = df[target_name]
        X = df.drop([target_name], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if ds == 'Scene':
            ffeatures = [f for f in ffeatures if
                         f not in ["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"]]
        else:
            ffeatures.remove(target_name)
        with open(os.path.join(out_dir,'clasiffiers_scores.txt'), 'a') as fout:
            fout.write('\n')
            fout.write(ds)
            fout.write('\n')
            fout.write('shape of training dataset: (%s,%s)' % (X_train.shape))
            fout.write('\n')
            fout.write('shape of testing dataset: (%s,%s)' % (X_test.shape))
            fout.write('\n')
        # load optimal params
        # optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
        for cls_method in ['logit', 'xgboost']:
            print(cls_method)
            optimal_params_filename = os.path.join(params_dir,
                                                   "optimal_params_%s_%s.pickle" %(cls_method, file_name))
            with open(optimal_params_filename, "rb") as fin:
                args[cls_method] = pickle.load(fin)

            if cls_method == 'logit':
                # Logistic Regression
                sscaler = StandardScaler()
                X_train = sscaler.fit_transform(X_train)
                X_test = sscaler.fit_transform(X_test)
                clf = LogisticRegression(C=2 ** args['logit']['C'], random_state=random_state)
            elif cls_method == 'xgboost':
                clf = xgb.XGBClassifier(objective='binary:logistic',
                                        n_estimators=500,
                                        learning_rate=args['xgboost']['learning_rate'],
                                        subsample=args['xgboost']['subsample'],
                                        max_depth=int(args['xgboost']['max_depth']),
                                        colsample_bytree=args['xgboost']['colsample_bytree'],
                                        min_child_weight=int(args['xgboost']['min_child_weight']),
                                        seed=random_state)
            elif cls_method == 'rf':
                        clf = RandomForestClassifier(n_estimators=500,
                                                        max_features=args['rf']['max_features'],
                                                        random_state=random_state, n_jobs=-1)


            elif cls_method == 'gbm':
                        clf = GradientBoostingClassifier(n_estimators=500,
                                           learning_rate=args['gbm']['learning_rate'],
                                           random_state=random_state)

            model_file = os.path.join(models_folder,
                                      'unfitted_model_%s_%s.pickle' %(cls_method, file_name))
            with open(model_file, 'wb') as fout:
                pickle.dump(clf, fout)
            clf.fit(X_train, y_train)
            model_file = os.path.join(models_folder,
                                      'model_%s_%s.pickle' %(cls_method, file_name))
            with open(model_file, 'wb') as fout:
                pickle.dump(clf, fout)
            if cls_method == 'xgboost':
                XGBoost_features = clf.get_booster().feature_names
                names = {'ffeatures': ffeatures, 'XGBoost_features': XGBoost_features}

            y_pred = clf.predict(X_test)
            acc = roc_auc_score(y_test, y_pred)
            pres = f1_score(y_test, y_pred, average='binary')
            confusion_mat = confusion_matrix(y_test, y_pred)
            #model_aic = calculate_abic(X_train.shape[0], pres, X_train.shape[1], 'AIC')
            #model_bic = calculate_abic(X_train.shape[0], pres, X_train.shape[1], 'BIC')
            with open(os.path.join(out_dir, 'clasiffiers_scores.txt'), 'a') as fout:
                fout.write('\n')
                fout.write('classifier arguments: \n')
                fout.write(str(args[cls_method]))
                fout.write('*********')
                fout.write('\n')
                fout.write('score of %s is: %s' % (cls_method, acc))
                fout.write('\n')
                fout.write('F1 score of %s is: %s' % (cls_method, pres))
                fout.write('\n')
                fout.write('confusion matrix of %s is:\n %s' % (cls_method, confusion_mat))
                fout.write('\n')
                '''out.write('AIC of %s is:\n %s' % (cls_method, model_aic))
                fout.write('\n')
                fout.write('BIC of %s is:\n %s' % (cls_method, model_bic))
                fout.write('\n')'''
                fout.write('--------------------------------------------------------------')
            for xai_method in ['perm', 'shap', 'ALE']:
                explain_predictions(xai_method, file_name, clf, ffeatures, X_train, y_train, cls_method)

        del args
