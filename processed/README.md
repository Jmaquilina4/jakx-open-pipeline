📂 Files

ALLJAK_bio_curated_ic50.csv

Combined curated IC50 → pIC50 assay data across JAK1, JAK2, JAK3, and TYK2.

Rows: 17,218 assay measurements

Unique compounds: 8,562

ALLJAK_bio_selectivity_ic50.csv

Compound × target selectivity matrix for JAK1–JAK2–JAK3–TYK2.

Used to evaluate full-panel inhibitor coverage.

Panel coverage: 1,097 compounds (12.8%)

ALLJAK_bio_panel_coverage_ic50.csv

Summary of per-target compound coverage.

Useful for panel completeness analysis.

JAK1_training_ic50.csv

JAK1-specific subset extracted from curated data.

Rows: 4,819 assay measurements

Compounds: 4,819 (before filtering)

JAK1_training_ic50_druglike.csv

Final QSAR-ready JAK1 dataset after applying strict drug-likeness filters.

Rows: 3,460 curated, drug-like compounds.

Filters applied: canonicalization, duplicate removal, salt stripping, PAINS and Brenk alerts.

ALLJAK_pipeline_summary_ic50.csv

Final statistics and reductions at each curation stage.

Useful for tracking reproducibility.
