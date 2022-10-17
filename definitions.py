import pandas as pd
import ast
import string
import os
from preprocessing.datasets_preprocessing import conc_dfs, preprocessing
from sklearn.preprocessing import KBinsDiscretizer

datasets= ['TruckFailure','Bank','Kidney','Credit','Heart','Spam','Climate', 'Adult','BreastC' ,'Diabetic','Ionosphere','spect']
cols ={ 'BreastC':[ 'codeNum','ClumpThickness','UniCellSize','UniCellShape',\
                  'MarginalAd','SingleEpithelialCellSize','BareNuclei',\
                  'BlandChromatin','NormalN','Mitoses', 'class'],\
       
       'Bank':['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',\
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',\
       'previous', 'poutcome', 'y'],\
       
       'Kidney':['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',\
       'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad',\
       'appet', 'pe', 'ane', 'class'],\
       
       'Credit':['b', '30.83', '0', 'u', 'g', 'w', 'v', '1.25', 't', 't.1', '01', 'f',\
       'g.1', '00202', '0.1', '+'],\
       
       'Heart':['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'],\
       
       'Spam':['Feat'+str(i) for i in range(1,58)]+['class'],\
       
       'TruckFailure':['class']+['Feat'+str(i) for i in range(1,171)],\

       'Climate':['Study', 'Run', 'vconst_corr', 'vconst_2', 'vconst_3', 'vconst_4',\
       'vconst_5', 'vconst_7', 'ah_corr', 'ah_bolus', 'slm_corr',\
       'efficiency_factor', 'tidal_mix_max', 'vertical_decay_scale',\
       'convect_corr', 'bckgrnd_vdc1', 'bckgrnd_vdc_ban', 'bckgrnd_vdc_eq',\
       'bckgrnd_vdc_psim', 'Prandtl', 'outcome'],
       
       'Adult': ['age', 'workclass', 'fnlwgt','education','education-num','marital-status',\
        'occupation','relationship','race','sex','capital-gain','capital-loss',\
        'hours-per-week','native-country','salary'],\
       
       'Diabetic':['Feat'+str(i) for i in range(1,20)]+['class'],\
       
       'Ionosphere':['Feat'+str(i) for i in range(1,35)]+['class'],\
       
       'spect':['F%sR' %(str(i)) for i in range(1,23)]+['class']}

targets_names = {'Bank':'y', 'Kidney':'class', 'Credit':'+',\
                 'Heart':'num', 'Spam':'class', 'TruckFailure':'class',\
                 'Climate':'outcome', 'Adult':'salary', 'BreastC':'class',\
                 'Diabetic':'class', 'Ionosphere':'class', 'spect':'class'}

files_names = {'Bank':'bank-full', 'Kidney':'chronic_kidney_disease_full', 'Credit':'crx',\
               'Heart':'processed.cleveland','Spam':'spambase',\
               'TruckFailure':'TruckFailures', 'Climate':'pop_failures', 'Adult':'adult',\
               'BreastC':'breast-cancer-wisconsin', 'Ionosphere':'ionosphere',\
               'Diabetic':'Diabetic_Retinopathy_Debrecen', 'spect':'SPECT'}

files_ends = {'Bank':'.csv', 'Kidney':'.csv', 'Credit':'.data', 'Heart':'.data',\
             'Spam':'.data', 'TruckFailure':'.csv', 'Climate':'.dat', 'Adult':'.csv',\
              'BreastC':'.data', 'Diabetic':'.arff', 'Ionosphere':'.data', 'spect':'.data'}

target_encoding = {'Heart':{0:0, 1:1, 2:1, 3:1, 4:1}, 'BreastC':{2:0, 4:1}}

datasets_info = pd.DataFrame(columns=['Dataset_name', 'Columns', 'targets_names', 'files_names', 'files_ends', 'target_encoding'])
datasets_info['Dataset_name'] = datasets
for key in cols.keys() & targets_names.keys() & files_names.keys() & files_ends.keys(): #& target_encoding.keys():  
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

datasets_info.to_csv(os.path.join(logs_dir, 'datasets_info.csv'), sep=';', index=False)
output_txt = os.path.join(logs_dir,'output.txt')

##########################################################################################################################################
