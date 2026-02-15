## Cardio Vascular Risk Assessment Prediction Model

### Author: Chandan Chaudhari

---

### Dataset Overview

This dataset is designed to assess and predict cardiovascular disease (CVD) risk using demographic, clinical, lifestyle, and behavioral features. It supports three analytical objectives:

1. **Risk Score Prediction** (Supervised Regression)

2. **Risk Category Classification** (Supervised Classification)

3. **Pattern Discovery** (Unsupervised Clustering)

Each row represents a unique patient observation, identified by `Patient_ID`.

---

### Feature Description

```
| Feature Name                     | Type                            | Description                                                           |
| -------------------------------- | ------------------------------- | --------------------------------------------------------------------- |
| Patient_ID                       | Categorical (Identifier)        | Unique identifier for each patient. Not used for modeling.            |
| age                              | Numerical (Continuous)          | Age of the patient in years.                                          |
| bmi                              | Numerical (Continuous)          | Body Mass Index, an indicator of body fat based on height and weight. |
| systolic_bp                      | Numerical (Continuous)          | Systolic blood pressure (mmHg).                                       |
| diastolic_bp                     | Numerical (Continuous)          | Diastolic blood pressure (mmHg).                                      |
| cholesterol_mg_dl                | Numerical (Continuous)          | Total cholesterol level in mg/dL.                                     |
| resting_heart_rate               | Numerical (Continuous)          | Resting heart rate (beats per minute).                                |
| smoking_status                   | Categorical (Binary/Ordinal)    | Smoking behavior (e.g., Non-smoker, Former smoker, Current smoker).   |
| daily_steps                      | Numerical (Continuous)          | Average number of steps walked per day.                               |
| stress_level                     | Numerical (Ordinal)             | Self-reported stress level on a defined scale (e.g., 1–10).           |
| physical_activity_hours_per_week | Numerical (Continuous)          | Weekly hours of physical exercise.                                    |
| sleep_hours                      | Numerical (Continuous)          | Average sleep duration per night (hours).                             |
| family_history_heart_disease     | Categorical (Binary)            | Indicates presence of heart disease in immediate family (Yes/No).     |
| diet_quality_score               | Numerical (Ordinal)             | Composite score representing overall diet quality.                    |
| alcohol_units_per_week           | Numerical (Continuous)          | Average alcohol consumption per week.                                 |
| heart_disease_risk_score         | Numerical (Continuous – Target) | Computed cardiovascular risk score (used for regression).             |
| risk_category                    | Categorical (Target)            | Risk classification label (e.g., Low, Medium, High).                  |

```
---

### Target Variables

1. **heart_disease_risk_score**

   * Used for regression modeling.
   * Represents a continuous risk index derived from clinical and lifestyle indicators.

2. **risk_category**

   * Used for classification modeling.
   * Derived from the risk score using predefined thresholds.

---

### Data Assumptions

* No direct data leakage: target variables are not used during feature engineering.

* `risk_category` is derived **only after** risk score computation.

* Lifestyle variables are self-reported and may contain mild noise.

* Dataset is assumed to be cross-sectional (single observation per patient).

---

### Applicability Across ML Approaches

* **Regression**: Predict continuous cardiovascular risk score.

* **Classification**: Predict discrete risk levels for clinical decision support.

* **Clustering**: Identify hidden patient segments based on health and lifestyle patterns.

This dataset structure is suitable for end-to-end ML, MLOps, and teaching-oriented implementations following best practices.

---

### Model Structure Designed


```
project_name/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── artifacts/
│       └── model.pkl
│
├── logs/
│
├── src/
│   ├── __init__.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── model_building.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── training_pipeline.py
│   │
│   ├── config.py
│   ├── logger.py
│   ├── exceptions.py
│   └── utils.py
│
├── serving/
│   ├── __init__.py
│   │
│   ├── main.py
│   │
│   ├── routes/
│   │   └── prediction_routes.py
│   │
│   ├── schemas/
│   │   └── prediction_schema.py
│   │
│   ├── services/
│   │   └── prediction_service.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       ├── css/
│       │   └── styles.css
│       └── js/
│           └── script.js
│
├── metrics.json
│
├── main.py
│
├── dvc.yaml
├── dvc.lock
│
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── requirements.txt
│
└── README.md

```

---

### To run the Pipeline:

- python main.py