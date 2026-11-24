# Census Income (Adult) — Salary Prediction with PySpark and MLlib

Predicting whether a person earns more than **US$50,000 a year** from U.S. Census data, using **PySpark** and **Spark MLlib** on a **Hadoop HDFS** cluster.

**Author:** Juan Sebastián Sánchez Arboleda
**Course:** Procesamiento de Alto Volumen de Datos — Pontificia Universidad Javeriana
**Deliverable:** `SanchezSalario_1255.ipynb`

---

## Overview

This project trains and compares **eight MLlib classifiers** to predict the binary income label (`<=50K` / `>50K`) of the UCI *Adult* (Census Income) dataset. The whole pipeline runs on Spark: the data is read from HDFS, prepared with a Spark `Pipeline`, and every model is trained and evaluated in MLlib. Pandas is used only to render the plots.

The work follows the **CRISP-DM** method and is organized as a single, fully documented Jupyter notebook written in Spanish.

## Dataset

- **Source:** [UCI Machine Learning Repository — Census Income (Adult)](https://archive.ics.uci.edu/dataset/20/census+income)
- **Files:** `adult.data` (32,561 rows) and `adult.test` (16,281 rows) — 48,842 records, 14 attributes plus the income label.
- The data is **not committed** to this repository; download it from the link above and upload it to HDFS (see below).

## Environment

- Apache Spark **3.5.5** (course cluster) with **Hadoop HDFS**
- JupyterHub, kernel **rapKernel_v4** (the notebook calls `findspark` when it detects the server)
- The notebook also runs outside the cluster: place `adult.data` and `adult.test` next to it, or point the `ADULT_DIR` environment variable at them.

Local dependencies (only needed off the cluster):

```bash
pip install pyspark==3.5.5 setuptools "pandas<3" matplotlib seaborn scikit-learn
```

## How to run on the cluster

1. Upload `SanchezSalario_1255.ipynb`, `adult.data` and `adult.test` to JupyterHub.
2. Put the two data files into HDFS (one time):
   ```bash
   hadoop fs -mkdir -p /Formatos1255/SanchezSalario
   hadoop fs -put adult.data adult.test /Formatos1255/SanchezSalario/
   ```
3. Open the notebook with the **rapKernel_v4** kernel.
4. **Kernel → Restart Kernel and Run All Cells.**

## Pipeline

1. **Data understanding** — raw inspection with an RDD, data dictionary, missing values (`?`), descriptive statistics, categorical and numeric plots, income rate per group, correlation matrix.
2. **Data preparation** — missing values kept as a `Desconocido` category (justified against mode imputation), country grouped into a `RegionOrigen` field, redundant columns dropped, and a Spark `Pipeline` of `StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler`. The pipeline is fit on training data only (no leakage).
3. **Modeling** — eight MLlib classifiers on the dataset's original train/test split.
4. **Evaluation** — accuracy, precision, recall, F1, ROC AUC and PR AUC against a majority-class baseline, confusion matrices, ROC and precision–recall curves, a second training round with class weights, and GBT feature importance.

## Models

Logistic Regression · Decision Tree · Random Forest · Gradient-Boosted Trees · Linear SVM · Naive Bayes · Factorization Machines · Multilayer Perceptron.

## Results (test set)

| Model | Accuracy | ROC AUC | PR AUC | F1 (>50K) |
| --- | --- | --- | --- | --- |
| **Gradient-Boosted Trees** | **0.861** | **0.913** | **0.786** | **0.680** |
| Multilayer Perceptron | 0.856 | 0.907 | 0.773 | 0.672 |
| Logistic Regression | 0.853 | 0.904 | 0.762 | 0.658 |
| Random Forest | 0.851 | 0.903 | 0.767 | 0.613 |
| Factorization Machines | 0.853 | 0.903 | 0.759 | 0.656 |
| Linear SVM | 0.853 | 0.901 | 0.754 | 0.653 |
| Decision Tree | 0.847 | 0.879 | 0.737 | 0.602 |
| Naive Bayes | 0.585 | 0.855 | 0.596 | 0.518 |
| Baseline (all ≤50K) | 0.764 | 0.500 | 0.236 | 0.000 |

**Gradient-Boosted Trees** is the best model. Its test error (~13.9%) is in line with the reference errors (14.05%–21.42%) reported in the dataset's `adult.names`. Training the models with **class weights** raises GBT recall from 62.4% to 85.2%, at the cost of precision — a decision that depends on the relative cost of false positives and false negatives.

## Repository structure

```
.
├── README.md
└── SanchezSalario_1255.ipynb   # full notebook (documentation, code and outputs)
```

## References

- Becker, B. & Kohavi, R. (1996). *Adult (Census Income)*. UCI Machine Learning Repository.
- Apache Spark 3.5.5 — [Classification and regression](https://spark.apache.org/docs/3.5.5/ml-classification-regression.html)
- Apache Spark 3.5.5 — [CSV Files](https://spark.apache.org/docs/3.5.5/sql-data-sources-csv.html)
