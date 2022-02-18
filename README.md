# ConsisXAI
The code in this repository is an implementation of the technique proposed in the paper with the title: "FeCoXAI: Features Consistency-based Metrics for Evaluating Explainable Artificial Intelligence". FeCoXAI is an implementation of a technique to evaluate global machine learning explainability (XAI) methods based on feature subset consistency.

To be able to replicate the paper findings and results, follow this execution order:

1- Run the code in "definitions.py" to transform the data files from the datasets_files folder into a format to be used in our experiments.

2- From the "modeling" folder, run "params_optimisation.py" to obtain the best params for each ML model to be used.

3- Run "main.py" which includes models training, explanations generation, and explanations evaluations using our proposed approach.

-- "experiments.py" can be used to compute consistency ratios using the number of features in the intersection subset instead of their scores. This experiment is discussed in the paper.


<img src=“https://github.com/GhadaElkhawaga/ConsisXAI/main/Proposed_approach2.jpg” alt="Proposed Approach">
![Proposed Approach](/GhadaElkhawaga/ConsisXAI/main/Proposed_approach2.jpg)

