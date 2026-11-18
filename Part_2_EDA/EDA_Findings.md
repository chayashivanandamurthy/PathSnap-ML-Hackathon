# PathSnap EDA Findings

## Scope and partition use

The supplied archive contains 80 training trips and 10 student-testing trips. Detailed data-quality checks, statistics, plots, network connectivity analysis, and maps in this contribution use the **training** partition. The **student-testing** partition is inventoried separately for trip/file availability and record counts only; it is not used to make training or preprocessing decisions. No evaluator-testing folder was present in the supplied archive, and none was accessed. The original extracted files were not modified.

The five supplied guideline PDFs were reviewed before analysis. File interpretation follows `Dataset Exploration.pdf`: files are headerless whitespace-separated records; track columns are longitude, latitude, and timestamp in seconds; node coordinates are indexed by row; arcs contain from/to node IDs and are indexed by row for arc IDs; route values are ordered arc IDs. The audit confirmed that all parsed records followed these layouts.

## Inventory

All four expected files were available for all 80 training trips (320/320 files) and all 10 student-testing trips (40/40 files). The scan found zero missing files, malformed records, or missing parsed cells.

Training record totals were 208,417 GPS observations, 1,933,485 node rows, 3,876,904 arc rows, and 31,336 reference-route arc IDs. Student-testing inventory totals were 19,160 GPS observations, 90,689 nodes, 185,041 arcs, and 3,477 route IDs. Inventory and per-file details are in `outputs/tables/partition_inventory.csv`, `file_inventory.csv`, `file_schemas.csv`, and `sample_records.csv`.

The exact duplicate-row audit counted repeated node coordinate pairs and repeated ordered arc endpoint pairs beyond the first occurrence, separately for each training trip. Across the 80 trips there were **89 duplicate node rows** (34 trips had at least one) and **1,775 duplicate arc rows** (53 trips had at least one). These rows were retained because row positions are node/arc IDs and later arc/route references depend on those positions. Per-trip counts and the overall row are in `outputs/tables/training_duplicates_sampling.csv`.

## Training GPS observations

- GPS observations per trip: median 1,529; range 361 to 19,313.
- Trip duration, calculated from the first and last recorded timestamps in file order: median about 1,710 seconds (28.5 minutes); range 451 to 154,030 seconds.
- Across 208,337 consecutive timestamp intervals, **201,053 were one second (96.503742%)** using `numpy.isclose(interval, 1.0, rtol=0, atol=1e-6)`. The remaining 7,284 intervals were not within that one-second tolerance. The overall percentage is pooled across intervals, not an unweighted average of the per-trip percentages. The interval median, 90th percentile, and 95th percentile were each 1 second; the 99th percentile was 3 seconds. There were 1,216 intervals greater than the notebook's descriptive 5-second gap threshold; the largest interval was 153,064 seconds.
- No training trip had non-increasing timestamps in file order. There were no out-of-range longitude/latitude values and no duplicate full GPS rows. Six consecutive pairs repeated the same coordinate.
- Duplicate node/arc cross-check for trip `00000001`: 8 repeated node coordinate rows and 292 repeated ordered arc endpoint rows. This matches the teammate's reported counts. Its 1,314 intervals comprise 1,289 one-second intervals and 25 other intervals (98.097412% one second), also matching. The earlier summary reported this trip's interval median and maximum, not the exact one-second/other split; it did not use a different interval definition. The audit uses consecutive timestamps in file order and the stated 1e-6-second absolute tolerance.
- Summing Haversine distances between consecutive valid GPS records gives an approximate path length with median 24.07 km, mean 37.23 km, and range 6.33 to 169.25 km. Large time gaps and GPS jumps can make this observed-point path estimate differ from the actual driven distance.
- Training trip bounds span approximately longitude -100.419 to 137.237 and latitude -34.640 to 61.267. The broad extent suggests the supplied trips are not confined to one small local area; coordinate calculations should respect geographic units and location.

## Road network and reference routes (training)

- Per-trip road-network sizes: median 9,608 node rows and 19,526.5 arc rows; ranges are 299–155,003 nodes and 649–307,392 arcs.
- Reference-route size: median 280 arc IDs; range 59–2,266.
- No arc endpoint referenced a node outside its trip's node table. No route arc ID fell outside its trip's arc table.
- The undirected projection of valid arcs had no isolated nodes. Component counts had median 20 (range 1–421). This summarizes weak connectivity only; direction-specific reachability can differ.
- Consecutive route IDs were checked by comparing the first arc's to-node with the next arc's from-node. Thirteen pairs were disconnected across 10 trips (up to three flagged pairs in a trip). This is a continuity warning, not proof that the supplied reference is wrong: the guideline cautions that route labels should not be treated as independently verified physical ground truth.

## Report-ready EDA summary

The raw dataset is organized by trip and provides a time-ordered GPS trajectory, trip-local node coordinates, directed road arcs, and a reference route encoded as ordered arc IDs. The inventory found complete four-file coverage in the supplied training and student-testing partitions. Training GPS records are generally sampled at one-second intervals, although long gaps are present, including an extreme 153,064-second interval. The coordinate distributions cover a broad geographic area. GPS path lengths were therefore calculated with Haversine distance rather than Euclidean distance in degrees.

Road-network sizes vary substantially between trips. Valid arc endpoints and route arc IDs were in range for all training trips. The undirected network projection had no isolated nodes but often contained multiple connected components. A small number of reference-route transitions did not connect end-to-start. These observations support checking each trip's graph IDs, preserving temporal order, and treating very long observation gaps and route continuity as explicit cases for Member 1's later preprocessing/feature-engineering work. This notebook did not implement those steps.

## Recommendations for Member 1

1. Preserve a per-trip boundary throughout reading and later splits; do not mix observations from the same trip across partitions.
2. Keep the raw timestamp order and add diagnostics for long sampling gaps. The 5-second threshold here is descriptive only; choose any operational threshold based on the team's task design and document it. Avoid silently connecting or interpolating across extreme gaps.
3. Treat coordinates as longitude/latitude, validate ranges and coordinate order, and use a geodesic distance or suitable local projection for physical distances.
4. Use row positions as node and arc IDs only within the corresponding trip. Check every arc endpoint and route arc ID before downstream use.
5. Check route transitions for end-to-start continuity; the supplied route IDs were not continuous at 13 adjacent pairs in 10 training trips.
6. Keep student-testing and evaluator-testing data out of any learned preprocessing fit. The evaluator partition was absent from this supplied archive.
7. Do not equate a GPS row index with a route row index. Route sequences and observations have different lengths and represent different objects.

## Generated figures and maps

Graphs in `outputs/graphs/`: `training_trip_distributions.png`, `training_sampling_intervals.png`, `training_geographic_extent.png`, `training_network_sizes.png`, `training_network_quality.png`, and `representative_training_trip_network_route.png`.

Interactive maps in `outputs/maps/`: `training_trip_00000022_map.html` (representative network/route view) and `training_trip_00000090_map.html` (trip with the largest observed timestamp gap). Both show GPS observations, local road-network context, and supplied reference-route arcs in separate layers.

CSV tables are stored under `outputs/tables/`; key files are `training_trip_summary.csv`, `training_duplicates_sampling.csv`, `training_quality_flags.csv`, `training_network_metrics.csv`, and `quality_overview.csv`.

## Assumptions and limitations

- File names and headerless column meanings follow the official exploration guide and were checked for parse validity and ID bounds.
- Arc row position is treated as arc ID, as specified by the guide. The graph component summary uses an undirected projection; route continuity uses directed end-to-start node IDs.
- A GPS gap greater than five seconds is flagged for descriptive analysis, not declared invalid or removed.
- Duplicate node/arc counts are exact repeated row values beyond the first copy: coordinate pairs for nodes and ordered endpoint pairs for arcs. These counts do not imply that the corresponding ID rows can safely be removed.
- One-second intervals use an absolute tolerance of 1e-6 seconds and zero relative tolerance to account for floating-point timestamp representation.
- Haversine trajectory length sums consecutive valid observations in file order. It can include straight-line jumps across gaps and is not a road-distance estimate.
- Road arcs have no supplied curved geometry in these files; static/map depictions connect endpoint coordinates with straight segments. Folium maps crop background network context to a small area around the GPS path for readability; all supplied route arcs are drawn.
- Reference routes are supplied reference information and may have limitations; they are not claimed as independently verified truth.
- Student-testing inventory is reported separately. The EDA does not compare training and student-testing distributions or use student-testing results to choose preprocessing.

## AI assistance record

AI assistance was used to draft the notebook structure, code, and report text. Inventory totals, quality flags, statistics, plots, and maps were generated by executing the notebook against the supplied raw data. The student author must verify this draft, understand the methods and findings, and make the disclosure required by the event's policy. No model experimentation, preprocessing pipeline, feature engineering, route decoder, backend, or frontend is claimed as Part 2 work.
