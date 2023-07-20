# Overview
TBI Patient Endotype Clustering - using LCA to demonstrate the existence of patient endotypes in a TBI population that differ in their comorbidity profiles and clinical outcomes.

# Methodology
## Core data sources
- Requires MIMIC III v1.4 database to be built in SQLite format locally
1. `tbi2_admit_icd.xlsx` - contains all TBI patient demographic and comorbidity information (e.g., DOB, sex, comorbidity ICDs, discharge status)
	- Query found in `MIMIC3 Linked TBI Desktop v2.accdb` MS Access file under `tbi_admit_icd` query
	- Note that the `.accdb` file must be manually linked to the locally built SQLite MIMIC III database in order for the query to be run
2. `tbi2_admit_proced.xlsx` - contains neurosurgical intervention events for TBI patients
	- Full query found in `sql_queries.py`
3. `tbi2_admit_chevents_gcs.xlsx` - contains recorded GCS scores for TBI patients
	- Full query found in `sql_queries.py`

## Data analysis pipeline
1. The core data sources are combined and processed by `processing.py` to generate `tbi2_admit_icd_dates_nsx_gcs_elix.xlsx` containing all key clinical information of interest
2. Using ICD-9 data from `tbi2_admit_icd_dates_nsx_gcs_elix.xlsx`, comorbidity endotypes are identified using `cluster.r` and appended for each patient onto the original Excel file to create `tbi2_admit_icd_dates_nsx_gcs_elix_annotated.xlsx`
3. Statistical analyses and visulizations are generated using data from `tbi2_admit_icd_dates_nsx_gcs_elix_annotated.xlsx` using the following files:
	- `graph_builder.py` and `graph_renderer.py` for comorbidity network graphs and relative risk between comorbidities
	- `visulization_stats.py` for conventional descriptive statistics and stratified analyses (by GCS and age)
	- `cluster.r` for visulization of comorbidity distributions 
	- `visulization_stats.r` for jitter plot and ANCOVA and regression analysis for endotypes

# Demo

![Survival jitter plot](/demo/survival_jitter.png "Survival jitter plot")
![Comorbidity network graph](/demo/networkmap.png "Comorbidity network graph")
![Comorbidity endotypes stability](/demo/LCA5x30_LPA13xVEI.png "Comorbidity endotypes stability")
![Comorbidity endotypes consolidated](/demo/LCA_5.png "Comorbidity endotypes consolidated")