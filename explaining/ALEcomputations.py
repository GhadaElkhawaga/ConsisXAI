import random
from math import log

def refine_probabilities(l):
  vals = []
  for x in range(len(l)):
        v = l[x]
        if v < 0:
          v = 1 - l[x]
        vals.append(v)
  return vals


def get_counts(l1,l2):
  count_c0, count_c1 = 0, 0
  for x in range(len(l1)):
    if l1[x] > l2[x]:
      count_c0 += 1
    elif l1[x] == l2[x]:
      selected = random.randint(0,1)
      if selected == 0:
        count_c0 += 1
      else:
        count_c1 += 1
    else:
      count_c1 += 1
  return count_c0, count_c1


def compute_entropy(l):
  return -1 * (sum([v*log(v,2) for v in l if v!=0]))


def feats_impurity(feature_names, ale_values):
  ent = []
  for feat in range(len(feature_names)):
    vals_c0, vals_c1, probs  = ([] for i in range(3))
    ent_feat = 0
    cat_count = len(ale_values[feat][:,0])
    vals_c0 = refine_probabilities(ale_values[feat][:,0])
    vals_c1 = refine_probabilities(ale_values[feat][:,1])
    for x in get_counts(vals_c0, vals_c1):
        probs.append(x/cat_count)
    for y in probs:
        if y == 0:
          continue
        else:
          ent_feat += (y*(log(y,2)))
    ent_feat = ent_feat * -1
    #check correctness of computations:
    if ent_feat > log(cat_count, 2):
        print('Maybe there is an error in calculations')
    ent.append(ent_feat)
  return ent

