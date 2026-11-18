\# Part 3 – Machine Learning



\## Overview



This module implements the machine learning preparation and candidate-scoring components of PathSnap. It analyses GPS-to-road candidate features and prepares the workflow for supervised candidate classification and integration with the HMM/Viterbi route reconstruction module.



\## Project Structure



\- `machine\_learning/` – Data validation, feature analysis, preprocessing, evaluation, experiment logging, model configuration, comparison and prediction export.

\- `experiments/plots/` – Training and student-testing feature distribution visualisations.

\- `experiments/results/` – Feature reports, outlier investigation reports and model comparison structure.

\- `experiments/logs/` – Experiment tracking log.

\- `reports/` – Part 3 technical report.



\## Candidate Models



The model setup covers five candidate classifiers:



1\. Logistic Regression

2\. K-Nearest Neighbours

3\. Decision Tree

4\. Random Forest

5\. HistGradientBoosting



\## Evaluation Strategy



The evaluation utilities support accuracy, precision, recall, F1-score, log loss, confusion matrix, observation-level Top-1 accuracy and candidate miss rate.



Trip-aware cross-validation is planned using `trip\_id` as the grouping variable.



\## Hyperparameter Optimisation



Optuna TPE configuration is provided with a reproducible random seed and an initial search budget of 30 trials per model.



\## Current Implementation Status



Data validation, feature analysis, outlier investigation, preprocessing utilities, evaluation utilities, experiment logging, prediction export, model comparison structure and Bayesian optimisation configuration have been implemented.



Supervised model training, tuning and final performance evaluation remain pending validated reference labels and an approved environment in which the required compiled ML dependencies can run.



See \[reports/PART3\_ML\_REPORT.md](reports/PART3\_ML\_REPORT.md) for further details.

