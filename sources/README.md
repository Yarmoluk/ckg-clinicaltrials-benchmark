# Frozen Source Snapshots

`frozen-normalized/` contains the 12 ClinicalTrials.gov-derived `corpus.json`
files used to build the benchmark graphs and prose, plus their build summaries.
These files freeze every normalized field consumed by the graph builder.

The per-trial `source_hash` values were calculated from the complete API study
objects before extraction. Those complete original response objects were not
preserved during the September 8, 2026 build, so the historical per-trial hashes
cannot be recomputed from these normalized snapshots. Integrity-v2/v3 therefore
hashes this entire frozen normalized tree in its run manifest and does not claim
that it closes the earlier raw-API provenance gap.
