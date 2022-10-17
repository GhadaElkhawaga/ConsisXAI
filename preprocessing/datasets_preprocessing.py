import pandas as pd
import os
from utils.retrieval import retrieve_artefact
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour

def check_perf(output_txt, results, minority_class):
  original_recall = results['original'][1][minority_class]
  original_precision = results['original'][0][minority_class]
  original_f1score = results['original'][2][minority_class]
  best_recall, best_precision, best_f1score = original_recall, original_precision, original_f1score
  best_method_recall, best_method_precision, best_method_f1score = 'original', 'original', 'original'

  for k, v in results.items():
      if v[1][minority_class] > best_recall:
        best_recall = v[1][minority_class]
        best_method_recall = k
      if v[0][minority_class] > best_precision:
        best_precision = v[0][minority_class]
        best_method_precision = k
      if v[2][minority_class] > best_f1score:
        best_f1score = v[2][minority_class]
        best_method_f1score = k

  with open(output_txt, 'a') as f:
    f.write('\n best method is %s with the highest recall of %s' %(best_method_recall, best_recall))
    f.write('\n best method is %s with the highest precision of %s' %(best_method_precision, best_precision))
    f.write('\n best method is %s with the highest f1score of %s' %(best_method_f1score, best_f1score))

  return best_method_recall, best_recall, best_method_precision, best_precision, best_method_f1score, best_f1score
  
#############################################################################################################################################  
def oversample(X, y, method):
  if method == 'randomover':
    ovrmodel = RandomOverSampler(random_state=42) 
  else:
    ovrmodel = SMOTE(random_state=42)
  
  X_train_over, y_train_over = ovrmodel.fit_resample(X, y)
  # Check the number of records after over sampling
  return X_train_over, y_train_over
#####################################################################################################################################################
def undersample(X, y, method):
  if method == 'randomunder':
    undermodel = RandomUnderSampler(random_state=42) 
    X_train_under, y_train_under = undermodel.fit_resample(X, y)
  elif method == 'nearmiss':
    nearmiss = NearMiss(version=3)
    X_train_under, y_train_under = nearmiss.fit_resample(X, y)
  elif method == 'cnn':
    cnn = CondensedNearestNeighbour(n_neighbors=1)
    X_train_under, y_train_under = cnn.fit_resample(X, y)

  # Check the number of records after over sampling
  return X_train_under, y_train_under
##########################################################################################################################################  
def model_perf(X_train,y_train, X_test, y_test):
  rf = RandomForestClassifier()
  baseline_model = rf.fit(X_train, y_train)
  y_pred = baseline_model.predict(X_test)
  # Check the model performance
  return precision_recall_fscore_support(y_test, y_pred, average=None)
#######################################################################################################################################  
def rebalance(output_txt, X_train, y_train, X_test, y_test, minority_class):   
    methods = {'oversampling_methods':['randomover', 'smote'], 'undersampling_methods':['randomunder', 'nearmiss']}
    results = {key:[] for key in ['original', 'over_randomover', 'over_smote', 'under_randomunder', 'under_nearmiss']}
    sampled_data_X = {key:[] for key in ['over_randomover', 'over_smote', 'under_randomunder', 'under_nearmiss']}
    sampled_data_y = {key:[] for key in ['over_randomover', 'over_smote', 'under_randomunder', 'under_nearmiss']}

    #performance report after classification using the current dataset
    report = model_perf(X_train,y_train,X_test,y_test)  
    results['original'] = report    

    for k,v in methods.items():
      if k == 'oversampling_methods':
        for vv in v:
          results_key = 'over_%s' %vv
          sampled_data_X[results_key], sampled_data_y[results_key] = oversample(X_train, y_train, vv)         
          results[results_key] = model_perf(sampled_data_X[results_key],sampled_data_y[results_key],X_test,y_test)
      else:
        for vv in v:
          results_key = 'under_%s' %vv
          sampled_data_X[results_key], sampled_data_y[results_key] = undersample(X_train, y_train, vv)            
          results[results_key] = model_perf(sampled_data_X[results_key],sampled_data_y[results_key],X_test,y_test)
    
    results_str_list = ["{0}:{1}\n".format(k,v) for k,v in results.items()]
    with open(output_txt,'a') as f:
      f.write('\n')
      f.writelines(results_str_list)

    best_method_recall, best_recall, best_method_precision, best_precision, best_method_f1score, best_f1score = check_perf(output_txt, results, minority_class) 

    #balance based on the algorithm resulting in recall improvement, 
    #if none then balance based on precision improvement, if none, then return the imbalanced data and work on it
    if best_method_recall != 'original':
      sampled_train_X = sampled_data_X[best_method_recall]
      sampled_train_y = sampled_data_y[best_method_recall]
      best_method =  best_method_recall
    elif best_method_precision != 'original':
      sampled_train_X = sampled_data_X[best_method_precision]
      sampled_train_y = sampled_data_y[best_method_precision]
      best_method = best_method_precision
    elif best_method_f1score != 'original':
      sampled_train_X = sampled_data_X[best_method_f1score]
      sampled_train_y = sampled_data_y[best_method_f1score]
      best_method = best_method_f1score
    else:
      sampled_train_X = X_train
      sampled_train_y = y_train
      best_method = 'original'

    del sampled_data_X, sampled_data_y
    return sampled_train_X, sampled_train_y, best_method
#############################################################################################################################################   
def preprocessing(output_txt, ds_name, cols,target_name, filename, file_end, target_encoding= None):
  print(ds_name)
  
  df = retrieve_artefact(logs_dir, filename, file_end)
  df.columns = cols

  #drop duplicate rows if they exceed 5% of the total df length
  if ((len(df)- len(df.drop_duplicates()))/len(df)) > 0.05 : 
    df = df.drop_duplicates()

  if ds_name in ['Bank','Kidney',"Credit","TruckFailure", 'Adult', 'Ionosphere']:
    lbl = LabelEncoder()
    df[target_name] = lbl.fit_transform(df[target_name])

  if ds_name == 'Diabetic':
    df[target_name] = pd.to_numeric(df[target_name])
  
  if target_encoding:
    df[target_name].replace(target_encoding, inplace=True)

  df.convert_dtypes()
  if ds_name in ["TruckFailure"]:   
      df.replace('na','-1', inplace=True)
      
  df.replace('?', -1, inplace=True)
  df.convert_dtypes() 
  if ds_name in ['Kidney', 'Credit']:
    num_Cols = {'Kidney':['bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot','hemo','pcv','wbcc', 'rbcc'], 'Credit':['30.83', '00202']}
    df[num_Cols[ds_name]] = df[num_Cols[ds_name]].apply(pd.to_numeric)

  if ds_name in ['Bank', 'Adult', 'Kidney', 'Heart']:
    df['age'].replace('?',-1, inplace=True)
    df['age'] = df['age'].astype(int)
  
  if ds_name in ["TruckFailure", "Ionosphere"]: 
    df = df.apply(pd.to_numeric)

  objcols = list(df.select_dtypes(exclude=['float64', 'float32', 'int64','int32', 'int', 'float']).columns)  
  for objc in objcols:
    df[objc] = df[objc].astype(str)
  
  colnames = ["{} \n".format(i) for i in objcols]
  with open(output_txt, 'a') as f:
    f.write('\n')
    f.write(ds_name)
    f.write('\n')
    f.writelines(colnames)

  
  if min(df[target_name].value_counts(normalize=True).values) < 0.4:
    print('imbalanced')

  val_counts = ["{} \n".format(i) for i in df[target_name].value_counts(normalize=True).values]
  with open(output_txt, 'a') as f:
    f.write('\n')
    f.writelines(val_counts)
    f.write('\n')
    f.write(str(min(df[target_name].value_counts(normalize=True).values)))
    if min(df[target_name].value_counts(normalize=True).values) < 0.4:
       f.write('\n imbalanced')
  
  """
  #pandas profiles after cleaning
  profile = ProfileReport(df, title="Pandas Profiling Report")
  profile.to_file(os.path.join(logs_dir, "Report_%s_preprocessed.html" %fname))
  """
  if objcols:
    lbl_enc_cols = [x for x in objcols if df[x].nunique()>5]
    one_hot_cols =  list(set(objcols) - set(lbl_enc_cols))      
    for cc in lbl_enc_cols:
      df[cc] = df[cc].astype('category').cat.codes
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

  df = df.apply(pd.to_numeric)
      
    
  y = df[target_name]
  X = df.drop([target_name], axis=1)

  #splitting and checking imbalance and then rebalancing if one class has less than 40% of the samples
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  sampling_method = 'original'
  if min(Counter(y_train).values()) < int(math.floor(0.4*len(y_train))):
      #over or under sample the target
      for key, value in Counter(y_train).items():
        if min(Counter(y_train).values())  == value:
          minority_class = key
      
      with open(output_txt, 'a') as f:
        f.write('\n the minority class is: %s' %minority_class)  
      new_X_train, new_y_train, sampling_method = rebalance(output_txt, X_train, y_train, X_test, y_test, minority_class)
      X_train = new_X_train
      y_train = new_y_train
  
  return X_train, X_test, y_train, y_test, objcols, sampling_method


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
