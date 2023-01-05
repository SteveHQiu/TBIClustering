import os
import pandas as pd

from graph_builder import GraphBuilder
from graph_renderer import GraphVisualizer

BETTER_LABELS = ["CHF", "Arrhythmia", "Valvular disease", "Pulmonary circulation disorder",
                "Peripheral vascular disorder", "Uncomplicated hypertension",
                "Complicated hypertension", "Paralysis", "Other neurological disorder",
                "COPD", "Uncomplicated diabetes", "Complicated diabetes", "Hypothyroidism",
                "Renal failure", "Liver disease", "Peptic ulcer disease", "AID/HIV",
                "Lymphoma", "Metastatic cancer", "Solid tumor (no metastasis)",
                "Rheumatoid arthritis", "Coagulopathy", "Obesity", "Weight loss",
                "Fluid and electrolyte disorders", "Blood loss anemia", "Deficiency anemia",
                "Alcohol abuse", "Drug abuse", "Psychoses", "Depression"]

DF_PATH = "data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4_BL.xlsx"
DF_ROOT_NAME = os.path.splitext(DF_PATH)[0]
DF = pd.read_excel(DF_PATH)
COL_SUB = "Endotype"

if 0:
    a = GraphBuilder()
    a.buildGraphLogistic(DF_PATH, cols_logistical=BETTER_LABELS)
    a.exportGraph()

    b = GraphVisualizer(F"{DF_ROOT_NAME}_logi.xml")
    b.genRenderArgs()
    b.genLegend()
    b.renderGraphNX(title="", cmap=False)

if 1:
    for endotype in DF[COL_SUB].unique():
        a = GraphBuilder()
        a.buildGraphLogistic(DF_PATH, cols_logistical=BETTER_LABELS,
                            col_sub=COL_SUB, subset=endotype)
        a.exportGraph()

        b = GraphVisualizer(F"{DF_ROOT_NAME}_logi_{COL_SUB}_{endotype}.xml")
        b.genRenderArgs()
        b.genLegend()
        # b.renderGraphNX(title=F"Endotype {endotype}", cmap=False)
        b.renderGraphNX(title=F"Endotype {endotype}", cmap=True)
