import numbers
from math import log

ef calculate_abic(n, ratio, num_params,calctype):
  ''' ratio is the reduct ratio or core ratio values, n is the number of 
  samples in the dataset, num_params is the length of the intersection set'''

  ratio = 1 - ratio #computing based on the complement
  if ratio >= 0 :
    print(ratio)
    if calctype == 'AIC':     
      val = -2 * log(ratio,2) + 2 * num_params 
    elif calctype == 'BIC':
      val = -2  * log(ratio,2) + num_params * log(n,2) 
    else:
      raise ValueError('Undefined calculation type')
  else: 
    val = 'Undefined'

  return val


def Compute_dissimilarity(original, xai_ratio):
  if xai_ratio > 0 and isinstance(xai_ratio, numbers.Number):
    min_val = min(original/xai_ratio, xai_ratio/original)
  else:
    min_val = 'Undefined'
  return min_val

def get_consistency_val(computed_ratio, dataset_ratio):
  if isinstance(computed_ratio, numbers.Number): 
      val = Compute_dissimilarity(dataset_ratio, computed_ratio)
  else:
      val = 'Undefined'  
  return val
  
