# PathSnap – Part 1: Feature Engineering and Candidate Generation

## 1. Dataset Description

The PathSnap project uses GPS trajectory data and road network information for map matching and route reconstruction.

### 1.1 Training Dataset
- Number of trips: 80
- GPS observations: 208,417
- Road network nodes: 1,933,485
- Road network arcs: 3,876,904
- Route IDs: 31,336

### 1.2 Student Testing Dataset
- Number of trips: 10
- GPS observations: 19,160
- Road network nodes: 90,689
- Road network arcs: 185,041
- Route IDs: 3,477

Each trip folder contains:
- `.track` – GPS trajectory observations.
- `.nodes` – Road network node information.
- `.arcs` – Road segments connecting nodes.
- `.route` – Ordered sequence of reference road arc IDs.

The `.route` file is not directly aligned one-to-one with GPS observations. Reliable GPS-to-route alignment is required before generating supervised learning labels.

## 2. GPS Features Explanation

The following GPS features are generated:

| Feature | Description |
|---|---|
| longitude | Longitude coordinate of the GPS observation. |
| latitude | Latitude coordinate of the GPS observation. |
| timestamp | Date and time of the GPS observation. |
| time_gap | Time difference between consecutive GPS observations. |
| distance_m | Distance travelled between consecutive GPS observations, in metres. |
| speed_mps | Estimated speed in metres per second. |
| speed_kmh | Estimated speed in kilometres per hour. |
| bearing_deg | Direction of movement between consecutive GPS observations, in degrees. |

Speed is calculated using:

`speed_mps = distance_m / time_gap`

`speed_kmh = speed_mps * 3.6`

Bearing represents the direction of movement between consecutive GPS observations. The first observation of each trip may have a missing bearing because no previous observation is available.

## 3. Candidate-Specific Features Explanation

Two candidate-specific features are generated for every GPS observation and its candidate road arc.

### 3.1 GPS-to-Road Distance
Feature: `gps_to_road_distance_m`

This represents the shortest distance between a GPS observation and a candidate road segment, measured in metres. A smaller value indicates that the GPS point is closer to the candidate road.

### 3.2 Bearing Difference
Feature: `bearing_difference_deg`

This represents the angular difference between the GPS movement direction and the candidate road direction, measured in degrees. A smaller difference indicates better directional alignment.

This feature may be missing for the first observation of each trip because its GPS movement bearing is unavailable.

## 4. Top 10 Candidate Generation Methodology

The candidate-generation pipeline identifies up to 10 possible road arcs for each GPS observation.

### Step 1: Load Dataset
Load the GPS track, road network nodes and road network arcs for each trip.

### Step 2: Coordinate Projection
Use a local Azimuthal Equidistant (AEQD) projection to represent geographical coordinates in metres for local distance calculations.

### Step 3: Build Spatial Search Structure
Build a KD-tree (`cKDTree`) using projected road network node coordinates for efficient nearest-node searches.

### Step 4: Find Nearby Nodes
Initially search for the 30 nearest road network nodes for each GPS observation. Expand the search to 100 nodes if the initial search does not provide enough adjacent candidate arcs.

### Step 5: Identify Candidate Arcs
Collect road arcs connected to the selected nearby nodes.

### Step 6: Calculate Candidate Features
Calculate GPS-to-road distance and bearing difference for each candidate arc.

### Step 7: Rank Candidates
Rank candidate road arcs based on their proximity to the GPS observation and retain up to 10 candidates.

### Step 8: Generate CSV Files
Save GPS features and candidate-specific features in separate training and student-testing CSV files.

## 5. Generated Candidate Datasets

| Output File | Trips | GPS Observations | Candidate Rows |
|---|---:|---:|---:|
| training_candidates.csv | 80 | 208,417 | 2,084,170 |
| student_testing_candidates.csv | 10 | 19,160 | 191,600 |

Each GPS observation has 10 candidate rows in the generated datasets. Candidate ranks range from 1 to 10.

## 6. Candidate CSV Columns

### Identification and Ranking
- `trip_id`
- `observation_id`
- `candidate_arc_id`
- `candidate_rank`
- `timestamp`

### GPS Features
- `longitude`
- `latitude`
- `time_gap`
- `distance_m`
- `speed_mps`
- `speed_kmh`
- `bearing_deg`

### Candidate-Specific Features
- `gps_to_road_distance_m`
- `bearing_difference_deg`

### Reference and Label Fields
- `reference_arc_id`
- `is_correct_candidate`

The reference and label fields are currently blank because reliable GPS-to-route alignment has not yet been completed. They must not be used as supervised training targets until validated labels are available.

## 7. Current Limitations and Pending Work

1. Candidate generation uses nearby road network nodes and their adjacent arcs as a heuristic.
2. Top 10 candidate recall has not yet been measured. The correct road arc is not guaranteed to appear in every candidate set.
3. GPS-to-route reference alignment must be validated before generating supervised labels.
4. Top 5 versus Top 10 candidate recall validation is pending reliable ground-truth labels.
5. Final model integration, route reconstruction and API integration will be documented after the respective project components are completed.

## 8. Part 1 Deliverables

- GPS feature extraction.
- Candidate-specific feature generation.
- Top 10 candidate road arc generation.
- Training candidate dataset for 80 trips.
- Student testing candidate dataset for 10 trips.
- Data quality report.
- Candidate dataset validation.

This document describes the current Part 1 implementation. Label-generation methodology and evaluation results will be added after validation and coordination with the other project components.
