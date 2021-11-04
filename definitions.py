import pandas as pd
import ast
import string
import os
from preprocessing.datasets_preprocessing import conc_dfs, preprocessing
from sklearn.preprocessing import KBinsDiscretizer

training_sizes, testing_sizes, total_bins = [], [], []
datasets_folder = 'datasets_files'
datasets = ['BreastC', 'Wine', 'Zoo', 'Diabetic','Ionosphere', 'spect', 'Vehicle', 'Scene']
Vehicle_files = [os.path.join(datasets_folder, 'xa'+str(i)+'.dat')
                 for i in string.ascii_lowercase[:9]]
Vehicle_header = [item.lower() for item in ['COMPACTNESS', 'CIRCULARITY',
                                            'DIST_CIRCULARITY', 'RADIUS RATIO',
                  'PRAXIS_ASPECT_RATIO', 'MAX_LENGTH_ASPECT_RATIO', 'SCATTER_RATIO',
                  'ELONGATEDNESS', 'PR_AXIS_RECTANGULARITY_AREA',
                                            'MAX_LENGTH_RECTANGULARITY_AREA',
                  'SCALED_VARIANCE_major', 'SCALED_VARIANCE_minor', 'SCALED_RADIUS_GYRATION',
                  'SKEWNESS_ABOUT_major', 'SKEWNESS_ABOUT_minor', 'KURTOSIS_ABOUT_minor',
                  'KURTOSIS_ABOUT_major', 'HOLLOWS_RATIO','class']]

conc_dfs(datasets_folder, Vehicle_files, Vehicle_header)
cols = {'BreastC': ['codeNum', 'ClumpThickness', 'UniCellSize', 'UniCellShape',
                  'MarginalAd', 'SingleEpithelialCellSize', 'BareNuclei',
                  'BlandChromatin', 'NormalN', 'Mitoses', 'class'],
       'Wine': ['class', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityAsh', 'Magnesium',
               'Totalphenols', 'Flavanoids', 'Nonflavanoidphenols',
               'Proanthocyanins', 'Colorintensity', 'Hue', 'diluted wine', 'Proline'],
       'Zoo': ['name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic',
              'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins',
              'legs', 'tail', 'domestic', 'catsize', 'type'],
       'Diabetic': ['Feat'+str(i) for i in range(1, 20)]+['class'],
       'Ionosphere': ['Feat'+str(i) for i in range(1,35)]+['class'],
       'Scene': ['Feat'+str(i) for i in range(1,295)]+["Beach", "Sunset",
                                                      "FallFoliage", "Field", "Mountain", "Urban"],
       'spect': ['F%sR' %(str(i)) for i in range(1, 23)]+['class'],
       'Vehicle': Vehicle_header
       }
targets_names = {'BreastC': 'class', 'Wine': 'class', 'Zoo': 'type',
                 'Diabetic': 'class', 'Ionosphere': 'class', 'Scene': 'target',
                 'spect': 'class', 'Vehicle': 'class'}
files_names = {'BreastC': 'breast-cancer-wisconsin', 'Wine': 'wine', 'Zoo': 'zoo',
               'Diabetic': 'Diabetic_Retinopathy_Debrecen', 'Ionosphere': 'ionosphere',
               'Scene': 'Scene', 'spect': 'SPECT', 'Vehicle': 'Vehicles'}
files_ends = {'BreastC': '.data', 'Wine': '.data', 'Zoo': '.data', 'Diabetic': '.arff',
             'Ionosphere': '.data', 'Scene': '.csv', 'spect': '.data', 'Vehicle': '.csv'}
target_encoding = {'BreastC': {2: 0, 4: 1}, 'Wine': {1: 0, 2: 1, 3:1},
                   'Zoo': {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1},
              'Scene': {1: 0, 2:1, 3: 1}, 'Vehicle': {1: 0, 2: 1, 3: 1}}
datasets_info = pd.DataFrame(columns=['Dataset_name', 'Columns', 'targets_names',
                                      'files_names', 'files_ends', 'target_encoding',
                                      'Training_size', 'Testing_size'])
datasets_info['Dataset_name'] = datasets
for key in cols.keys() & targets_names.keys() & files_names.keys() & files_ends.keys():
    datasets_info.loc[datasets_info['Dataset_name'] == key, 'Columns'] = [cols[key]]
    datasets_info.loc[datasets_info['Dataset_name'] == key, 'targets_names'] = targets_names[key]
    datasets_info.loc[datasets_info['Dataset_name'] == key, 'files_names'] = files_names[key]
    datasets_info.loc[datasets_info['Dataset_name'] == key, 'files_ends'] = files_ends[key]
    if key in target_encoding.keys():
        datasets_info.loc[datasets_info['Dataset_name'] == key, 'target_encoding'] = str(target_encoding[key])
    else:
        datasets_info.loc[datasets_info['Dataset_name'] == key, 'target_encoding'] = 'Already_encoded'
for i in datasets_info['target_encoding']:
    if i != 'Already_encoded':
        i = ast.literal_eval(i)
for ds in datasets:
    if ds in ['Diabetic', 'Ionosphere', 'spect']:
        XTrain, XTest, yTrain, yTest, updated_cols = preprocessing(ds, cols[ds], targets_names[ds],
                                                                   os.path.join(datasets_folder, files_names[ds]),
                                                                   files_ends[ds])
    else:
        XTrain, XTest, yTrain, yTest, updated_cols = preprocessing(ds, cols[ds], targets_names[ds],
                                                                   os.path.join(datasets_folder, files_names[ds]),
                                                                   files_ends[ds], target_encoding[ds])
    X_df = pd.concat([XTrain, XTest], axis=0)
    y_series = pd.DataFrame(pd.concat([yTrain, yTest], axis=0), columns=[targets_names[ds]])
    for i in X_df.columns:
        if len(X_df[i].value_counts()) >= 10:
            bins = 4
        else:
            bins = len(X_df[i].value_counts())
        total_bins.append(bins)

    df = pd.concat([X_df, y_series], axis=1, join='inner')
    df.to_csv(os.path.join(datasets_folder, 'preprocessed_not_discretized_dataset_%s.csv'
                           %(files_names[ds])), sep=',', index=False)
    discretizer = KBinsDiscretizer(n_bins=min(total_bins), encode='ordinal', strategy='uniform')
    header = list(updated_cols)
    header.remove(targets_names[ds])
    X_df = pd.DataFrame(discretizer.fit_transform(X_df), columns=header)

    df = pd.concat([X_df, y_series], axis=1, join='inner')
    df.to_csv(os.path.join(datasets_folder, 'discretized_dataset_%s.csv' %(files_names[ds])),
              sep=',', index=False)
    training_sizes.append(XTrain.shape)
    testing_sizes.append(XTest.shape)

datasets_info['Training_size'] = training_sizes
datasets_info['Testing_size'] = testing_sizes
datasets_info.to_csv(os.path.join(datasets_folder, 'datasets_info.csv'), sep=',', index=False)
