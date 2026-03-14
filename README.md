# Data Validation with TFDV — Bank Marketing Dataset

**MLOps Course Lab | Northeastern University | Spring 2026**

A complete end-to-end data validation pipeline using [TensorFlow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started) applied to the UCI Bank Marketing dataset. This lab demonstrates how to generate statistics, infer schemas, detect anomalies, handle schema environments, check for data drift/skew, and perform data slicing — all essential components of a production MLOps data pipeline.

---

## Modifications from Original Lab

This is a modified version of the **TFDV Data Validation Lab** (originally using the Diabetes 130-US Hospitals dataset). The following changes were made to demonstrate independent understanding of the concepts:

### Dataset Change

| Attribute | Original (Diabetes) | Modified (Bank Marketing) |
|-----------|---------------------|---------------------------|
| Source | UCI Diabetes 130-US Hospitals | UCI Bank Marketing |
| Records | ~101,766 | ~45,211 |
| Features | 50 | 17 |
| Categorical Features | Medications, diagnoses | Job, marital, education, month, poutcome |
| Numerical Features | Encounter details | Age, balance, duration, campaign, pdays |
| Missing Values | Encoded as `?` | Encoded as `unknown` |
| Label | `readmitted` (multiclass) | `y` — subscribed to term deposit (binary) |
| Domain | Healthcare / Hospital readmission | Finance / Direct marketing campaigns |

### Custom Additions (5 new sections)

1. **Missing Value Analysis** — Visual heatmap and bar chart analyzing `unknown`-encoded missing data patterns across features, identifying that `poutcome` (81.7%), `contact` (28.8%), and `education` (4.0%) have significant missing rates.

2. **Feature Correlation Heatmap** — Triangular correlation matrix of all numerical features with automatic detection of highly correlated pairs (|r| > 0.5). Helps identify redundant features before model training.

3. **Anomaly Summary Report** — A reusable `generate_anomaly_report()` function that produces a structured pandas DataFrame of all detected anomalies with severity, description, and affected features. Designed for integration into automated MLOps monitoring pipelines.

4. **Multi-feature Data Slicing** — Beyond the original single-feature slicing, this lab slices by:
   - `education` level (4 slices: primary, secondary, tertiary, unknown)
   - `marital` status (3 slices: married, single, divorced) — for fairness analysis
   - `job` type (12 slices: admin, blue-collar, entrepreneur, etc.)

5. **Data Quality Score Dashboard** — A custom `compute_data_quality_score()` function that evaluates data across four dimensions:
   - **Completeness** — % of non-null, non-unknown values
   - **Consistency** — % of features passing schema validation
   - **Uniqueness** — Diversity ratio of categorical feature values
   - **Validity** — % of numerical values within expected ranges

   Includes bar chart comparison across splits and a radar chart for the quality profile.

---

## TFDV Pipeline Overview

The lab follows the standard TFDV data validation workflow:

```
Load Data → Split (70/15/15) → Generate Statistics → Infer Schema
    ↓                                                      ↓
Visualize Stats ← Compare Train vs Eval ← Detect Anomalies
    ↓                                                      ↓
Fix Anomalies → Configure Environments → Drift/Skew Check
    ↓                                                      ↓
Data Slicing → Quality Dashboard → Freeze Schema (.pbtxt)
```

### Step-by-step Breakdown

| Step | Section | Description |
|------|---------|-------------|
| 1 | Load & Split | Download UCI Bank Marketing CSV, split into train (70%), eval (15%), serve (15%) with shuffling |
| 2 | Generate Stats | Use `tfdv.generate_statistics_from_dataframe()` to compute descriptive statistics |
| 3 | Visualize | Interactive Facets-based visualization of feature distributions |
| 4 | Infer Schema | `tfdv.infer_schema()` to auto-generate expected feature types, domains, and constraints |
| 5 | Detect Anomalies | `tfdv.validate_statistics()` to find mismatches between eval data and training schema |
| 6 | Fix Anomalies | Expand schema domains, relax distribution constraints, add missing categorical values |
| 7 | Environments | Define TRAINING and SERVING environments; exclude label column from SERVING schema |
| 8 | Drift & Skew | Set L-infinity norm thresholds for `marital` (skew), `job` (drift), `housing` (skew) |
| 9 | Data Slicing | Compute per-slice statistics for education, marital status, and job type |
| 10 | Freeze Schema | Export validated schema to `output/schema.pbtxt` for production use |

---

## Dataset

**UCI Bank Marketing Dataset**

The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict whether the client will subscribe to a term deposit (`y`).

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Citation**: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31.
- **Records**: 45,211
- **Features**: 17 (7 numerical + 10 categorical)

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Client age in years |
| `job` | Categorical | Type of job (12 categories: admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown) |
| `marital` | Categorical | Marital status (married, divorced, single) |
| `education` | Categorical | Education level (primary, secondary, tertiary, unknown) |
| `default` | Binary | Has credit in default? (yes/no) |
| `balance` | Numeric | Average yearly balance in euros |
| `housing` | Binary | Has housing loan? (yes/no) |
| `loan` | Binary | Has personal loan? (yes/no) |
| `contact` | Categorical | Contact communication type (cellular, telephone, unknown) |
| `day` | Numeric | Last contact day of the month (1-31) |
| `month` | Categorical | Last contact month (jan-dec) |
| `duration` | Numeric | Last contact duration in seconds |
| `campaign` | Numeric | Number of contacts during this campaign |
| `pdays` | Numeric | Days since last contact from previous campaign (-1 = not contacted) |
| `previous` | Numeric | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous campaign (failure, other, success, unknown) |
| `y` | Binary | **Target** — subscribed to term deposit? (yes/no) |

---

## Setup & Installation

### Option A: Google Colab (Recommended)

TFDV requires Python 3.9–3.11. Since Colab runs Python 3.12 by default, follow these steps:

**Step 1** — Install Python 3.11 (run once):
```python
!sudo apt-get update -y -qq
!sudo apt-get install -y -qq python3.11 python3.11-dev python3.11-venv python3.11-distutils
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
!sudo update-alternatives --set python3 /usr/bin/python3.11
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
```

Then go to **Runtime → Restart runtime**.

**Step 2** — Install packages (run after restart):
```python
!pip install tensorflow==2.17.0
!pip install tensorflow-data-validation==1.17.0
!pip install matplotlib seaborn
```

### Option B: Local Setup (Linux with conda)

```bash
conda create -n tfdv_lab python=3.11 -y
conda activate tfdv_lab
pip install tensorflow==2.17.0
pip install tensorflow-data-validation==1.16.0
pip install matplotlib seaborn jupyter
jupyter notebook TFDV_Bank_Marketing_Lab.ipynb
```

> **Note**: TFDV only publishes Linux x86_64 wheels. It does not support Windows or macOS ARM (Apple Silicon) natively.

---

## Project Structure

```
tfdv-bank-marketing-lab/
├── TFDV_Bank_Marketing_Lab.ipynb   # Main notebook with all exercises and custom additions
├── README.md                        # This file
└── output/
    └── schema.pbtxt                 # Frozen schema (generated by running the notebook)
```

---

## Key Results

### Data Split Sizes
- Training: ~31,647 records (70%)
- Evaluation: ~6,782 records (15%)
- Serving: ~6,782 records (15%, label dropped)

### Anomalies Detected & Fixed
- Categorical domain mismatches between train/eval splits (fixed by expanding domains)
- Label column (`y`) missing in serving data (fixed via schema environments)
- Distribution constraint violations in `poutcome` and `contact` (relaxed to 90% domain mass)

### Drift & Skew Analysis
- L-infinity norm thresholds set at 0.03 for `marital` (skew), `job` (drift), and `housing` (skew)
- With shuffled splits, detected distances are close to threshold — demonstrating TFDV's sensitivity with ~45k records

### Data Quality Scores
Automated quality scoring across both splits:
- **Completeness**: ~95% (some features have 'unknown' values)
- **Consistency**: 100% after schema fixes
- **Validity**: 100% for numerical features
- **Overall**: >95% across both training and evaluation

---

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| TensorFlow | 2.17.0 |
| TensorFlow Data Validation | 1.17.0 |
| Apache Beam | (TFDV dependency) |
| pandas | >= 1.5 |
| matplotlib | >= 3.7 |
| seaborn | >= 0.12 |

---

## References

- [TFDV Documentation](https://www.tensorflow.org/tfx/data_validation/get_started)
- [TFDV API Reference](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv)
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31.

---

## Author

**Malav Patel**
MS Computer Science, Northeastern University

- GitHub: [github.com/Malav2002](https://github.com/Malav2002)
- LinkedIn: [linkedin.com/in/malavxpatel](https://linkedin.com/in/malavxpatel)
- Email: patel.malav@northeastern.edu
