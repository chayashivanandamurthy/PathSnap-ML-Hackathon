# PathSnap – GPS-Based Map Matching and Route Reconstruction

<p align="center">
  <strong>Transforming Noisy GPS Observations into Meaningful Road Routes</strong>
</p>

## Overview

**PathSnap** is a GPS-based map-matching and route reconstruction project designed to identify the most likely road segments travelled by a vehicle using raw GPS observations.

GPS data collected from mobile devices and tracking systems can contain noise, location inaccuracies, irregular sampling intervals and signal disturbances. Consequently, directly connecting GPS coordinates may produce unrealistic routes or incorrectly matched road segments.

PathSnap addresses this challenge through a modular pipeline combining GPS data preprocessing, exploratory data analysis, candidate road generation, machine learning-based candidate scoring and Hidden Markov Model (HMM)-based route reconstruction.

The objective is to transform raw GPS trajectories into realistic and continuous road-network paths.

## Problem Statement

Raw GPS trajectories do not always accurately represent the roads travelled by a vehicle. Common challenges include:

- GPS measurement noise and positional inaccuracies.
- Irregular time intervals between observations.
- Multiple nearby road segments for a single GPS coordinate.
- Incorrect road assignments based only on geographical proximity.
- Discontinuous or unrealistic reconstructed routes.

PathSnap aims to address these challenges by considering both individual GPS observations and the relationships between consecutive road segments.

## Project Objectives

- Analyse and understand GPS trajectory datasets.
- Identify data-quality issues and inconsistencies.
- Generate multiple possible road-segment candidates for GPS observations.
- Extract meaningful spatial and movement-related features.
- Develop a machine learning framework for candidate scoring.
- Reconstruct likely road sequences using HMM and Viterbi.
- Visualise reconstructed routes using road-network information.

## System Architecture

PathSnap follows a modular architecture consisting of four major components.

### Part 1 – Data Preprocessing and Candidate Generation

This component prepares GPS observations and generates possible road-segment candidates.

Key responsibilities:
- Inspect and validate input GPS data.
- Process coordinates and timestamps.
- Extract movement-related information.
- Generate possible road-segment candidates.
- Prepare candidate-level datasets for subsequent processing.

### Part 2 – Exploratory Data Analysis (EDA)

This component analyses the available GPS and road-network datasets.

Key activities:
- Dataset inventory and schema analysis.
- Missing-value and duplicate inspection.
- GPS trajectory distribution analysis.
- Spatial and geographic extent analysis.
- Sampling interval analysis.
- Road-network quality investigation.
- Data visualisation and reporting.

### Part 3 – Machine Learning

This component prepares the candidate-scoring framework. The objective is to assign scores to possible road segments using GPS and road-related features.

**Features considered:**
- Candidate rank.
- Longitude and latitude.
- Time gap.
- Distance between GPS observations.
- Speed in metres per second and kilometres per hour.
- Bearing.
- GPS-to-road distance.
- Bearing difference.

**Candidate classification models:**
1. Logistic Regression
2. K-Nearest Neighbours (KNN)
3. Decision Tree
4. Random Forest
5. HistGradientBoosting

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Log Loss
- Confusion Matrix
- Observation-level Top-1 Accuracy
- Candidate Miss Rate

**Hyperparameter optimisation:**

Optuna with a Tree-structured Parzen Estimator (TPE) sampler is configured for Bayesian hyperparameter optimisation. Trip-aware cross-validation using `trip_id` as the grouping variable is planned to prevent observations from the same trip leaking across training and validation folds.

### Part 4 – Route Reconstruction

This component reconstructs a likely sequence of road segments using probabilistic sequence modelling.

Key components include:
- Road network processing.
- Candidate adaptation.
- Hidden Markov Model.
- Viterbi decoding.
- Route reconstruction pipeline.
- Map payload generation.
- Route visualisation support.

The HMM models the relationship between observed GPS positions and possible hidden road states. The Viterbi algorithm identifies the most likely sequence of road states.

## End-to-End Workflow

```text
          RAW GPS OBSERVATIONS
                   |
                   v
        DATA PREPROCESSING
                   |
                   v
       EXPLORATORY DATA ANALYSIS
                   |
                   v
        CANDIDATE GENERATION
                   |
                   v
       FEATURE EXTRACTION
                   |
                   v
        MACHINE LEARNING
       CANDIDATE SCORING
                   |
                   v
         HMM EMISSION SCORES
                   |
                   v
         VITERBI DECODING
                   |
                   v
        ROUTE RECONSTRUCTION
                   |
                   v
        ROUTE VISUALISATION
```

## Repository Structure

```text
PathSnap-ML-Hackathon/
│
├── Part_2_EDA/
│   ├── outputs/
│   │   ├── graphs/
│   │   ├── maps/
│   │   └── tables/
│   ├── PathSnap_EDA.ipynb
│   ├── EDA_Findings.md
│   ├── README.md
│   └── requirements.txt
│
├── Part_3_Machine_Learning/
│   ├── machine_learning/
│   ├── experiments/
│   │   ├── plots/
│   │   ├── results/
│   │   └── logs/
│   ├── models/
│   ├── reports/
│   ├── results/
│   ├── scripts/
│   ├── src/
│   ├── visualisations/
│   ├── README.md
│   └── requirements.txt
│
├── Part_4_Route_Reconstruction/
│   ├── candidate_adapter.py
│   ├── hmm_viterbi.py
│   ├── map_payload.py
│   ├── pipeline.py
│   ├── road_network.py
│   └── test_*.py
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
│
├── src/
│   ├── features/
│   └── preprocessing/
│
├── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| NumPy | Numerical computation |
| Pandas | Data processing |
| Scikit-learn | Machine learning |
| Optuna | Hyperparameter optimisation |
| Matplotlib / Seaborn | Data visualisation |
| HMM | Probabilistic route modelling |
| Viterbi Algorithm | Most likely route sequence |
| HTML, CSS, JavaScript | Frontend interface |
| Git and GitHub | Version control and collaboration |

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chayashivanandamurthy/PathSnap-ML-Hackathon.git
```

### 2. Navigate into the project

```bash
cd PathSnap-ML-Hackathon
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows:

```cmd
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Explore the project

Start by examining the individual module README files and their documented scripts.

**Note:** The exact execution entry point and required data paths depend on the selected pipeline and dataset configuration. Refer to the respective module documentation before running individual components.

## Dataset

The project uses GPS trajectory observations and road-network candidate information.

The machine learning preparation currently includes:

| Dataset | Trips | Candidate Rows | Observations |
|---|---:|---:|---:|
| Training | 80 | 2,084,170 | 208,417 |
| Student Testing | 10 | 191,600 | 19,160 |

Each observation initially contains ten road-segment candidates. Dataset availability and access permissions may vary. Large raw datasets are not included in this repository by default.

## Testing and Evaluation

The project includes utilities for:
- Dataset validation.
- Feature analysis.
- Outlier investigation.
- Candidate evaluation.
- Experiment tracking.
- Route reconstruction testing.

For Part 3, the evaluation and model comparison frameworks are prepared. Supervised training, Bayesian optimisation trials and final performance comparisons require validated reference labels and a working approved ML environment.

No model accuracy or final performance results are claimed until those experiments have been executed.

## Current Development Status

| Component | Development status |
|---|---|
| Data preprocessing and candidate generation | Implemented components |
| Exploratory Data Analysis | Analysis and reports available |
| Machine Learning | Preparation and evaluation framework implemented; supervised experiments pending |
| Route reconstruction | HMM/Viterbi components and tests available |
| Frontend | Interface files available |
| End-to-end integration | Requires final integration and validation |

## Future Enhancements

- Complete supervised model training and comparison.
- Perform Bayesian hyperparameter optimisation.
- Improve candidate-generation accuracy.
- Integrate validated ML candidate scores with HMM emissions.
- Improve route continuity and reconstruction accuracy.
- Expand evaluation across additional GPS trajectories.
- Improve interactive route visualisation.
- Optimise processing time for large GPS datasets.

## Team Members

- Chaya
- Ankita Vellara
- Danyl Bince
- Hamsa M N

## Team Contributions

PathSnap is developed collaboratively, with responsibilities divided across the project modules:
- **Part 1:** Data preprocessing and candidate generation.
- **Part 2:** Exploratory Data Analysis.
- **Part 3:** Machine Learning and candidate scoring.
- **Part 4:** HMM-based route reconstruction and Viterbi decoding.

## Conclusion

PathSnap demonstrates a modular approach to GPS map matching by combining spatial data processing, exploratory analysis, machine learning and probabilistic route reconstruction.

By considering multiple road candidates and their sequential relationships, the system aims to produce more meaningful and realistic road trajectories from noisy GPS observations.

---

**PathSnap – From Raw GPS Coordinates to Reconstructed Road Routes.**
