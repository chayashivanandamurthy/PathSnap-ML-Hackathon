# PathSnap Part 2: Dataset Exploration and EDA

This folder contains the Part 2 exploratory analysis only. It reads the supplied raw files without modifying them. It does not implement data preprocessing or feature engineering, model training, Bayesian optimization, HMM/Viterbi, FastAPI, or a frontend.

## Dataset location

The notebook expects the extracted source folder at:

```text
../dataset_source/Student folder/
```

From the project root, extract `Student folder.zip` into a folder named `dataset_source`. The supplied archive contains `train` and `student_test`; it does not contain evaluator-testing data. Do not add evaluator data to this EDA run. If the extracted dataset is in another location, edit `DATA_ROOT` in the setup cell of `PathSnap_EDA.ipynb`.

## Install and run on Windows

From this folder, create and activate an environment, install the listed packages, then start Jupyter:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook PathSnap_EDA.ipynb
```

Run the notebook from the first cell to the last. CSV tables, PNG graphs, and interactive Folium HTML maps are regenerated under `outputs/`. The maps use OpenStreetMap tiles, so a network connection may be needed to display the basemap in a browser.

## Output folders

- `outputs/tables/`: trip inventory, schemas, sample records, training-trip summaries, the per-trip duplicate/sampling audit with overall totals, data-quality flags, and network metrics.
- `outputs/graphs/`: six labelled PNG figures.
- `outputs/maps/`: two interactive maps of selected training trips.

Detailed descriptive EDA and maps use the training partition only. Student-testing trips are counted and inventoried separately, without using them to make training decisions. The notebook does not access evaluator-testing data.

## Academic integrity and AI assistance

AI assistance was used to draft notebook code and explanatory text. All reported counts, tables, and figures were computed by executing the notebook against the supplied files. The student author should review the results, understand and be able to explain each method, and disclose AI assistance in the format required by the course/hackathon policy before submission.
