# Heart Disease source data

Source: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/45/heart+disease

Citation: Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano (1989). *Heart Disease*. DOI: https://doi.org/10.24432/C52P4X.

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/.

The four `raw/processed.*.data` files are unmodified copies of the public processed site files downloaded for the 2026-10-10 revision. Each has 13 input columns and the original five-level `num` target; `?` denotes missing values. The experimental scripts convert `num > 0` to a binary label. No records are excluded. Sites contain 303, 294, 123, and 200 records respectively.

The package redistributes these files with attribution. Data are unchanged; transformations used in analysis are described in the manuscript and saved per-fold manifests. The original download and hashes are recorded separately under `exp/eswa/`.
