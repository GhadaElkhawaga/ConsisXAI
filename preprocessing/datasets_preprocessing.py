import pandas as pd
import os
from utils.retrieval import retrieve_artefact
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split


def preprocessing(out_dir, ds_name, cols, target_name, filename, file_end, target_encoding=None):
    print(ds_name)
    if file_end in ['.csv', '.data']:
        df = retrieve_artefact(filename, file_end, ',')
    else:
        df = retrieve_artefact(filename, file_end)
    df.columns = cols
    # drop duplicate rows if they exceed 5% of the total df length
    if ((len(df) - len(df.drop_duplicates())) / len(df)) > 0.05:
        df = df.drop_duplicates()
    if ds_name == 'BreastC':  # this dataset has 16 entries with '?' sign indicating missing values
        df.replace({'?': -1}, inplace=True)
        df = df.apply(pd.to_numeric)
        # df['BareNuclei'] = pd.to_numeric(df['BareNuclei'])
    if ds_name in ['Ionosphere', 'Vehicle']:
        lbl = LabelEncoder()
        df[target_name] = lbl.fit_transform(df[target_name])
    if ds_name == 'Scene':
        target_cols = ["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"]
        df[target_name] = df[target_cols].sum(axis=1)
        df = df.drop(target_cols, axis=1)
    if target_encoding:
        df[target_name].replace(target_encoding, inplace=True)
        # print(df[target_name].value_counts())
    if ds_name == 'Diabetic':
        df[target_name] = pd.to_numeric(df[target_name])
    objcols = list(df.select_dtypes(exclude=['float64', 'float32', 'int64', 'int32']).columns)
    lbl = LabelEncoder()
    for i in objcols:
        df[i] = lbl.fit_transform(df[i])
    y = df[target_name]
    X = df.drop([target_name], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    return X_train, X_test, y_train, y_test, df.columns


def scale_update(old_dict):
    scale_df = pd.DataFrame()
    for x in old_dict.keys():
      vals = old_dict[x]
      scale_df[x] = vals.values()
    for col in scale_df.columns.tolist():
        scale_df[col] = MinMaxScaler().fit_transform(scale_df[col].values.reshape(-1, 1))
    for i in range(len(old_dict.keys())):
      keys = list(list(old_dict.values())[i].keys())
      scaled_vals = list(scale_df[scale_df.columns[i]].values)
      unscaled_dict = list(old_dict.values())[i]
      dictionary = dict(zip(keys,scaled_vals))
      unscaled_dict = unscaled_dict.update(dictionary)
    return


def Discretize(df, filename):
    for i in df.columns[:-1]:
        if (len(df[i].value_counts()) >= 10):
            bins = 4
        else:
            bins = len(df[i].value_counts())
        df[i] = pd.cut(df[i], bins=bins)  # discretize
        print(df.dtypes)
        # transform the resulting discretized
        objcols = list(df.select_dtypes(include=['category']).columns)
        lbl = LabelEncoder()
        for i in objcols:
            df[i] = lbl.fit_transform(df[i])
    print(len(list(df.select_dtypes(include=['category']).columns)))
    return df.to_csv('discretized_dataset_%s.csv' % (filename), sep=',', index=False, header=False)


# a function to concatenate all the .dat vehicles_files
def conc_dfs(datasets_folder,Vehicle_files, Vehicle_header):
    df_total = pd.DataFrame()
    for vf in Vehicle_files:
        df = pd.read_csv(vf, sep='\\s', engine='python', header=None)
        df_total = pd.concat([df_total,df])
    return df_total.to_csv(os.path.join(datasets_folder,'Vehicles.csv'), header=Vehicle_header, sep=',', index=None)


def compute_class_counts(folder, datasets, info_df,out_dir):
    for ds in datasets:
        file_name = info_df.loc[info_df['Dataset_name'] == ds, 'files_names'].values[0]
        target_name = info_df.loc[info_df['Dataset_name'] == ds, 'targets_names'].values[0]
        df = retrieve_artefact(folder, 'preprocessed_not_discretized_dataset_%s' %(file_name),                               '.csv', ',')
        with open(os.path.join(out_dir, 'class_counts.txt'), 'a') as fout:
              fout.write('Dataset: %s' %(ds))
              fout.write('\n')
              fout.write('Dataset shape: %s' %(str(df.shape)))
              fout.write('\n')
              fout.write(str(df[target_name].value_counts().to_numpy()))
              fout.write('\n')
