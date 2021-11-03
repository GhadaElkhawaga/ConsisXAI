import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import hyperopt
from hyperopt import hp, tpe, hp, Trials, STATUS_OK, fmin
from hyperopt.pyll.base import scope
from utils.retrieval import retrieve_artefact
import ast
import pickle
import os

datasets_folder = 'datasets_files'
if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), datasets_folder)):
  os.makedirs(os.path.join(os.path.dirname(os.getcwd()), datasets_folder))
#to get the file from the absolute path up one level to the current
os.chdir(os.path.dirname(os.getcwd()))
info_df = retrieve_artefact(datasets_folder, 'datasets_info', '.csv', ',')
datasets = ['BreastC', 'Wine', 'Zoo', 'Diabetic', 'Ionosphere', 'spect', 'Vehicle', 'Scene']
params_dir = 'optimal_params'
if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), params_dir)):
  os.makedirs(os.path.join(os.path.dirname(os.getcwd()), params_dir))
print(os.path.abspath(params_dir))

def create_and_evaluate_model(args):
    global trial_nr
    trial_nr += 1
    if cls_method == "xgboost":
        cls = xgb.XGBClassifier(objective='binary:logistic',
                                n_estimators=500,
                                learning_rate=args['learning_rate'],
                                subsample=args['subsample'],
                                max_depth=int(args['max_depth']),
                                colsample_bytree=args['colsample_bytree'],
                                min_child_weight=int(args['min_child_weight']),
                                seed=random_state)
    elif cls_method == "logit":
        cls = LogisticRegression(C=2 ** args['C'],
                                 random_state=random_state)
    score = cross_val_score(cls, X, y, cv=5, n_jobs=-1).mean()
    for k, v in args.items():
        fout_all.write("%s;%s;%s;%s;%s;%s\n" % (trial_nr, file_name, cls_method, k, v, score))
    fout_all.flush()
    return {'loss': -score, 'status': STATUS_OK}  # , 'model': cls}


n_iter = 50
random_state = 42
space_args = {'xgboost': {'learning_rate': hp.uniform("learning_rate", 0, 1),
                          'subsample': hp.uniform("subsample", 0.5, 1),
                          'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                          'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                          'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))},
              'logit': {'C': hp.loguniform('C', -15, 15)}
              }


for ds in datasets:
    file_name = info_df.loc[info_df['Dataset_name'] == ds, 'files_names'].values[0]
    ffeatures = ast.literal_eval(info_df.loc[info_df['Dataset_name'] == ds, 'Columns'].values[0])
    target_name = info_df.loc[info_df['Dataset_name'] == ds, 'targets_names'].values[0]
    print(ds)
    df = retrieve_artefact(datasets_folder, 'preprocessed_not_discretized_dataset_%s'
                           %(file_name), '.csv', ',')
    print(df.shape)
    y = df[target_name]
    X = df.drop([target_name], axis=1)
    for cls_method in ['logit', 'xgboost']:
        print(cls_method)
        if cls_method == "logit":
            sscaler = StandardScaler()
            X = sscaler.fit_transform(X)
        trial_nr = 1
        trials = Trials()
        space = space_args[cls_method]
        fout_all = open(os.path.join(params_dir,
                                     "param_optim_all_trials_%s_%s.csv" %(cls_method, file_name)), "w")
        fout_all.write("%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "param", "value", "score"))
        best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
        fout_all.close()
        print('best: ', best)
        # write the best parameters
        best_params = hyperopt.space_eval(space, best)
        print('best params:', best_params)
        outfile = os.path.join(params_dir, "optimal_params_%s_%s.pickle" % (cls_method, file_name))
        with open(outfile, "wb") as fout:
            pickle.dump(best_params, fout)

