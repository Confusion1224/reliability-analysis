# Reliability Analysis

This repository contains functions for calculating inter-rater and intra-rater reliability using pandas dataframe and currently available libraries, essential for assessing the consistency of ratings across multiple raters or the same rater over time.

## Features

1. Kappa Statistics: For categorical data (nominal/ordinal)
    - Cohen's Kappa (2 raters)
    - Fleiss' Kappa (2+ raters)

2. Agreement Measures
    - Krippendorff Alpha
    - Scott's Pi
    - Spearman's Intrarater Correlation
    - Cronbach's Alpha

3. Intraclass Correlation (ICC)
    - ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k

## How to use the Functions

To effectively use the functions for calculating inter-rater and intra-rater reliability, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/Confusion1224/reliability-analysis.git
```

### 2. Install required packages
```bash
pip install -r pip install -r reliability-analysis/requirements.txt
```

### 3. Modify the system path and import functions
```python
import sys
sys.path.append('reliability-analysis')
from functions.metrics import *
```

### 4. Load Data
Use pandas to load your rating data into a DataFrame before calculating statistics. Wide format is ideal for most statistics.

```python
import pandas as pd

# Example rating data - wide format (common for kappa statistics)
ratings_wide = pd.DataFrame({
    'PatientID': [101, 102, 103, 104, 105],
    'RaterA': [3, 2, 4, 3, 2],          # 1-5 severity scale
    'RaterB': [4, 2, 4, 3, 3],          # 1-5 severity scale
    'RaterC': [3, 3, 4, 2, 2]           # 1-5 severity scale
}).set_index('PatientID')  # Using PatientID as index

```

## Statistical Measures
This package provides three categories of inter-rater reliability statistics:

1. Kappa Statistics: For categorical data (nominal/ordinal)
    - Cohen's Kappa (2 raters)
    - Fleiss' Kappa (2+ raters)

2. Agreement Measures
    - Krippendorff Alpha
    - Scott's Pi
    - Spearman's Intrarater Correlation
    - Cronbach's Alpha

##  Inter-Rater Reliability Metrics Comparison

| Metric                  | Description                                                                 | Raters | Missing Data | Output Statistics                          | Data Format          |
|-------------------------|-----------------------------------------------------------------------------|--------|--------------|--------------------------------------------|----------------------|
| **Cohen's Kappa**       | Agreement between 2 raters (nominal/ordinal)                                | 2      | ❌           | κ, p-value, z-score, CI, etc                             | Wide only            |
| **Fleiss' Kappa**       | Agreement for multiple raters (nominal/ordinal)                             | 2+     | ✅           | κ                                          | Wide only            |
| **Krippendorff's Alpha**| Flexible reliability coefficient (nominal, ordinal, interval, ratio)                           | 2+     | ✅           | α                            | Wide only            |
| **Scott's Pi**          | Agreement accounting for chance (nominal)                                   | 2      | ❌           | π                                      | Wide only            |
| **Spearman Correlation**| Consistency between raters                         | 2      | ❌           | r, p-value                                 | Wide only            |
| **Cronbach's Alpha**    | Internal consistency of raters (interval/ratio)                             | 2+     | ✅           | α, item variance, total variance                    | Wide only            |

## Kappa Statistics
Kappa statistics measure agreement between raters while accounting for chance agreement. Values range from -1 (perfect disagreement) to 1 (perfect agreement), with 0 indicating chance agreement.

Interpretation Guide for Cohen Kappa and Fleiss Kappa:

≤ 0: No agreement

0.01-0.20: Slight agreement

0.21-0.40: Fair agreement

0.41-0.60: Moderate agreement

0.61-0.80: Substantial agreement

0.81-1.00: Almost perfect agreement

1. <b>Cohen's Kappa</b>

Use for 2 raters with categorical data (nominal or ordinal). Return all statistical results by setting return_results=True.
```python
kappa = cohens_kappa_from_df(
    df=ratings_wide,
    rater_a='Clinician',
    rater_b='Researcher',
    return_results=False  # Set True for detailed output
)

print(f"Cohen's Kappa: {kappa:.3f}")
```

<b>2. Fleiss Kappa</b>

Use for 2+ raters with categorical data (nominal or ordinal).
```python
from rater_pandas.kappa import fleiss_kappa_from_df

kappa = fleiss_kappa_from_df(df=ratings_wide)
print(f"Cohen's Kappa: {kappa:.3f}")
```

## Agreement Measures
Agreement measures assess the consistency between raters. These can be used for both categorical and continuous data, depending on the measure.

## Reliability Metrics Comparison

This table summarizes various reliability metrics and their interpretations.

| Range         | Scott's Pi              | Krippendorff's Alpha    | Spearman's Intrarater Correlation | Cronbach's Alpha               |
|---------------|-------------------------|--------------------------|-----------------------------------|---------------------------------|
| **≤ 0**       | No agreement            | Poor reliability         | No correlation                    | Poor reliability            |
| **0.01 - 0.20** | Slight agreement        | Weak reliability         | Weak correlation                  | Poor reliability            |
| **0.21 - 0.40** | Fair agreement          | Moderate reliability     | Moderate correlation               | Poor reliability      |
| **0.41 - 0.60** | Moderate agreement      | Acceptable reliability   | Fair correlation                  | Poor reliability      |
| **0.61 - 0.80** | Substantial agreement    | Strong reliability       | Strong correlation                | Moderate reliability               |
| **0.81 - 1.00** | Almost perfect agreement | Almost perfect reliability | Perfect correlation               | Strong reliability               |

<b>1. Krippendorff's Alpha</b>

A robust measure for any number of raters, works with nominal, ordinal, interval, or ratio data.

```python
alpha = krippendorff_alpha_from_df(
    df=ratings_long,
    unit_col='PatientID',
    rater_col='Rater',
    score_col='Score',
    level_of_measurement='interval'  # or 'nominal', 'ordinal', 'ratio'
)

print(f"Krippendorff's Alpha: {alpha:.3f}")
```

<b>2. Scott's Pi</b>

Similar to Cohen's Kappa but assumes equal marginal distributions.

```python
pi = scotts_pi_from_df(
    df=ratings_wide[['RaterA', 'RaterB']],  # For 2 raters
    return_results=False
)

print(f"Scott's Pi: {pi:.3f}")
```

<b>3. Spearman's Intrarater Correlation</b>
Measures consistency of a single rater's scores over time (test-retest reliability).

```python
corr = spearman_intrarater_from_df(
    df=ratings_wide,
    rater_cols=['RaterA', 'RaterB', 'RaterC'],
    patient_id_col='PatientID'
)

print(f"Spearman's Intrarater Correlation: {corr:.3f}")
```
<b>4. Cronbach's Alpha</b>

Measures internal consistency between multiple raters.

```python
alpha = cronbach_alpha_from_df(
    df=ratings_wide,
    rater_cols=['RaterA', 'RaterB', 'RaterC']
)

print(f"Cronbach's Alpha: {alpha:.3f}")
```
