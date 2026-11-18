

\# PathSnap – Part 3: Machine Learning



\## 1. Objective



Develop a machine learning-based candidate scoring system

to identify the most likely road segment corresponding to

each GPS observation. The predicted candidate scores will

be provided to Part 4 for HMM/Viterbi-based map matching.



\## 2. Dataset



| Dataset | Trips | Candidate Rows | Observations |

|---|---:|---:|---:|

| Training | 80 | 2,084,170 | 208,417 |

| Student Testing | 10 | 191,600 | 19,160 |



Each observation initially contains 10 candidate road segments.



The reference labels were pending validation at the time

of this report. No supervised model training results are

claimed.



\## 3. Feature Engineering



The following features are prepared:



\- Candidate rank

\- Longitude and latitude

\- Time gap

\- Distance between GPS observations

\- Speed in metres per second

\- Speed in kilometres per hour

\- Bearing

\- GPS-to-road distance

\- Bearing difference



Identifiers such as trip ID, observation ID and candidate

arc ID are retained for grouping and output, not used as

ordinary predictive features.



\## 4. Data Analysis



Implemented:

\- Dataset structure and completeness validation

\- Feature distribution analysis

\- Missing-value analysis

\- Outlier investigation

\- Feature distribution visualisations



Initial outlier thresholds were used to identify

observations requiring further review. Flagged observations

were not automatically removed.



\## 5. Candidate Models



Five candidate classifiers were configured:



1\. Logistic Regression

2\. K-Nearest Neighbours

3\. Decision Tree

4\. Random Forest

5\. HistGradientBoosting



Actual model training and comparison remain pending.



\## 6. Evaluation Metrics



The evaluation framework supports:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Log Loss

\- Confusion Matrix

\- Observation-level Top-1 Accuracy

\- Candidate Miss Rate



No real model performance metrics are reported yet.



\## 7. Bayesian Hyperparameter Optimisation



Optuna TPE sampler configured with:



\- 30 initial trials per model

\- Random seed: 42

\- Five-fold GroupKFold strategy

\- Grouping variable: trip\_id

\- Optimisation objective: F1-score



Actual optimisation trials remain pending.



\## 8. Experiment Tracking



Implemented experiment logging to record:

\- Model name

\- Hyperparameters

\- Evaluation metrics

\- Training and inference time

\- Random seed

\- Experiment notes



\## 9. Prediction Export



A CSV prediction export utility was implemented to generate

candidate-level scores for integration with Part 4.



Expected fields:

\- trip\_id

\- observation\_id

\- candidate\_arc\_id

\- prediction\_score



The export utility was tested using synthetic data.

No actual model predictions have been generated.



\## 10. Current Limitations



\- Validated reference labels are awaited from Part 1.

\- Windows Application Control blocked a compiled

&#x20; scikit-learn dependency in the local environment.

\- Actual model training and Bayesian optimisation

&#x20; could not be completed in this environment.

\- Final performance comparison and model selection

&#x20; remain pending.



\## 11. Conclusion



The machine learning preparation pipeline, data analysis

utilities, evaluation framework, experiment logger,

prediction exporter, model comparison structure and

Bayesian optimisation configuration have been implemented.



The next stage requires validated candidate labels and

an approved working machine learning environment to

complete model training, tuning and final evaluation.



