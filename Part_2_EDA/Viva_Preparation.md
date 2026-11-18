# PathSnap EDA: Viva Preparation

## What did this notebook do?

It read the supplied files, checked their structure and IDs, summarized the training GPS/network/route data, saved simple plots and made two interactive maps. It did not clean or rewrite the source data and did not train any model.

## What are the four file types?

- `.track`: longitude, latitude, and time in seconds for each GPS observation.
- `.nodes`: longitude and latitude for road-network nodes. A row's position is its node ID.
- `.arcs`: the start and end node IDs for a directed road segment. A row's position is its arc ID.
- `.route`: an ordered list of arc IDs that describes the supplied reference route.

A GPS observation and a route arc are different things, so their rows should not be matched by row number.

## Important methods and libraries

- **Pandas** reads, organizes, summarizes, and exports tables.
- **NumPy** handles arrays and numeric calculations.
- **Matplotlib** makes the saved PNG charts.
- **NetworkX** counts connected components and isolated nodes in a simple graph.
- **Folium** creates HTML maps with selectable GPS, road, and route layers.
- The **Haversine formula** estimates distance over Earth's curved surface from longitude and latitude. It is more appropriate than measuring degrees as if they were flat kilometers.
- A **connected component** is a group of nodes that can be connected by edges when edge direction is ignored. This notebook uses that undirected view only as a basic network summary.

## Main observed results

There are 80 training trips and 10 student-testing trips in the supplied archive, and all four files exist for each trip. Detailed analysis uses training data. It found no malformed parsed rows, missing files, invalid coordinates, invalid arc endpoints, invalid route IDs, or out-of-order timestamps. Across the training trips, 201,053 of 208,337 intervals are one second (96.503742%) with a 1e-6-second absolute floating-point tolerance. Another 7,284 intervals are not one second, including 1,216 intervals longer than five seconds; the largest is about 42.5 hours. Thirteen neighboring reference-route arc pairs did not connect end-to-start across 10 trips.

The duplicate audit found 89 repeated node coordinate rows across 34 trips and 1,775 repeated ordered arc endpoint rows across 53 trips. A duplicate row means its coordinate pair or endpoint pair has appeared earlier in that same trip's file. The row still has its own index ID, so the audit counts it but does not remove or reorder it. For trip 00000001, the notebook found 8 duplicate node rows, 292 duplicate arc rows, 1,289 one-second intervals, and 25 other intervals; these match the teammate's findings. The overall one-second percentage is pooled by interval count, rather than averaging the per-trip percentages.

The route is called supplied reference information rather than guaranteed truth because the official guide warns that it may have limitations. Network components are counted without direction, so that result does not mean every road is reachable in both directions.

## Figures and maps

`training_trip_distributions.png` shows trip sizes, durations, GPS path-length estimates, and route lengths. `training_sampling_intervals.png` shows the cadence and long gaps. `training_geographic_extent.png` shows the coordinate spread. `training_network_sizes.png` compares nodes/arcs and route size. `training_network_quality.png` summarizes isolated nodes and route transitions. The representative static network figure and two Folium maps compare GPS observations, road segments, and route arcs.

## Caveats worth explaining

- The five-second threshold is a descriptive flag, not a rule that deletes data.
- One-second interval counts use `numpy.isclose` with zero relative tolerance and `atol=1e-6` seconds.
- Duplicate node/arc rows retain their original positions because those positions are IDs referenced by other records.
- Summing GPS-to-GPS Haversine distances can overstate travel when a large gap separates observations.
- Road segments are drawn as straight lines because the supplied files give endpoint coordinates, not detailed curved road geometry.
- The archive has no evaluator-testing partition, and the notebook does not access one.
- Student-testing trips were inventoried separately and were not used to fit or choose preprocessing.

## AI-use disclosure

AI assistance helped draft notebook code and report explanations. The counts and visuals are produced from actual data by running the notebook. Be ready to describe the code, confirm the reported results, and disclose AI assistance in the required course format.
