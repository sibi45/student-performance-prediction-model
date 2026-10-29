# 🎓 Student Performance Prediction

A machine learning regression project that predicts student Performance Index using study habits, previous scores, sleep duration, extracurricular participation, and practice question-paper activity.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Model-Decision_Tree-2E8B57?style=for-the-badge" alt="Decision Tree">
  <img src="https://img.shields.io/badge/Tuning-GridSearchCV-6A5ACD?style=for-the-badge" alt="GridSearchCV">
  <img src="https://img.shields.io/badge/Validation-5--Fold_CV-8A2BE2?style=for-the-badge" alt="Five-fold cross-validation">
  <img src="https://img.shields.io/badge/Git-Version_Control-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">
</p>

## 📌 Project Overview

This project demonstrates an end-to-end machine learning workflow for predicting a numerical student Performance Index from five input features.

Multiple regression algorithms were trained and evaluated. Their performance was compared using standard regression metrics, and a Decision Tree Regressor was subsequently tuned using `GridSearchCV` with five-fold cross-validation.

The selected estimator was saved as a reusable model artifact using `joblib` and tested on new student inputs.

## 🎯 Objectives

- Explore and prepare a structured student-performance dataset.
- Train and compare multiple regression algorithms.
- Evaluate models using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- Investigate overfitting and underfitting by comparing training and test scores.
- Tune Decision Tree hyperparameters using cross-validation.
- Save and reload the selected model.
- Generate predictions for new student inputs.
- Maintain a reproducible and version-controlled ML project.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Model training, evaluation, and hyperparameter tuning |
| Jupyter Notebook | Interactive development and experimentation |
| Joblib | Saving and loading the trained model |
| Git | Version control |
| GitHub | Source code hosting |

## 📂 Dataset

The project uses `Student_Performance.csv`.

### Input features

| Feature | Description |
|---|---|
| `Hours Studied` | Number of hours spent studying |
| `Previous Scores` | Student's previous score |
| `Extracurricular Activities` | Encoded extracurricular participation |
| `Sleep Hours` | Number of hours spent sleeping |
| `Sample Question Papers Practiced` | Number of practice question papers completed |

**Target variable:** `Performance Index`

The target is numerical, so this project is a **regression problem**, not a classification problem.

The dataset's original source and license should be verified before redistribution. Dataset size, missing-value handling, duplicate handling, and other preprocessing details should be documented according to the actual notebook.

## 🔬 Machine Learning Workflow

1. **Data loading and exploration:** Load the CSV and inspect its structure, columns, and values.
2. **Data preparation:** Separate the input features from the target and prepare categorical data in the numerical format used by the model.
3. **Train-test split:** Divide the data into training and test sets.
4. **Model training:** Train multiple regression algorithms.
5. **Model comparison:** Compare predictive performance using regression metrics.
6. **Overfitting analysis:** Compare training and test performance.
7. **Hyperparameter tuning:** Use `GridSearchCV` with five-fold cross-validation to evaluate Decision Tree configurations.
8. **Model evaluation:** Evaluate the selected estimator on the test set.
9. **Model serialization:** Save the trained estimator using `joblib`.
10. **Inference:** Load the saved model and generate predictions for new student inputs.

## 🤖 Models Evaluated

The following regression algorithms were trained and compared during experimentation:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- K-Nearest Neighbors (KNN) Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR)

### Model comparison results

The earlier model comparison produced the following R² scores:

| Model | Training R² | Test R² |
|---|---:|---:|
| Linear Regression | 0.988640 | 0.989055 |
| Ridge Regression | 0.988640 | 0.989055 |
| Linear SVR | 0.988636 | 0.989042 |
| ElasticNet | 0.986648 | 0.986969 |
| Lasso | 0.986748 | 0.986918 |
| Random Forest | 0.997463 | 0.986042 |
| KNN | 0.988810 | 0.983951 |
| Decision Tree | 0.999324 | 0.976092 |

These values are from the earlier model comparison, before the separate Decision Tree hyperparameter-tuning experiment. The tuned Decision Tree results are reported independently below.

**Observation:** Linear Regression had the highest test R² in the original comparison. Random Forest and the original Decision Tree showed larger gaps between training and test performance, suggesting potential overfitting.

The Decision Tree was selected for further hyperparameter experimentation; it was not the highest-scoring model in the original test-set comparison.

## 🌳 Decision Tree Hyperparameter Tuning

`GridSearchCV` was used to systematically evaluate combinations of Decision Tree hyperparameters with five-fold cross-validation.

### Search configuration

| Setting | Value |
|---|---|
| Estimator | `DecisionTreeRegressor` |
| Search method | `GridSearchCV` |
| Cross-validation folds | 5 |
| Scoring metric | R² |
| Candidate `max_depth` values | `5`, `10`, `15`, `None` |
| Candidate `min_samples_leaf` values | `1`, `5`, `10`, `20` |
| Candidate `min_samples_split` values | `2`, `10`, `20`, `50` |
| Total combinations | 64 |
| Total cross-validation fits | 320 |

### Best hyperparameters found

| Hyperparameter | Selected value |
|---|---:|
| `max_depth` | `10` |
| `min_samples_leaf` | `10` |
| `min_samples_split` | `2` |

**Best cross-validation R²:** `0.9833897401`

These are the best settings among the candidate combinations evaluated. A grid search does not guarantee a globally optimal configuration beyond the supplied search space.

## 📊 Tuned Decision Tree Evaluation

The selected Decision Tree was evaluated using the held-out test set.

| Metric | Result |
|---|---:|
| Cross-validation R² | 0.98339 |
| Test R² | 0.98491 |
| Test MAE | 1.87952 |
| Test RMSE | 2.35805 |

### Metric interpretation

- **R²:** Measures how well predictions explain variation in the target relative to a mean-prediction baseline. Higher is generally better.
- **MAE:** Measures the average absolute difference between predicted and actual Performance Index values. Lower is better.
- **RMSE:** Measures prediction error while penalizing larger errors more strongly. Lower is better.

The tuned Decision Tree achieved a test R² of approximately `0.9849`. This is strong performance on the current test set, but it does not establish that the Decision Tree is the best model overall.

The earlier model comparison showed a higher test R² for Linear Regression. For a final production-oriented model selection, compare candidates using a consistent cross-validation procedure and evaluate the selected model on a fresh, untouched test set. The existing test results have already been used for model comparisons during experimentation.

## 💾 Saved Model

The selected model is saved at:

```text
mlmodel/student_performance_model.pkl
```

The model artifact contains the fitted Decision Tree estimator and its learned structure. It allows predictions to be generated without retraining the model each time.

### Load the model

```python
import joblib

model = joblib.load("mlmodel/student_performance_model.pkl")

print("Model loaded successfully!")
```

### Generate a prediction

The following example uses the same five features and numerical encoding used during training:

```python
import pandas as pd
import joblib

model = joblib.load("mlmodel/student_performance_model.pkl")

new_student = pd.DataFrame([{
    "Hours Studied": 7,
    "Previous Scores": 85,
    "Extracurricular Activities": 1,
    "Sleep Hours": 7,
    "Sample Question Papers Practiced": 4
}])

prediction = model.predict(new_student)

print("Predicted Performance Index:", round(float(prediction[0]), 2))
```

The prediction is an estimate, not a guaranteed outcome. The categorical feature must use the same encoding as the training data.

## 📁 Project Structure

```text
studentperformace/
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
└── mlmodel/
    ├── student.ipynb
    ├── student_performance_model.pkl
    └── Student_Performance.csv
```

This reflects the current project layout. Update the structure if files are renamed, moved, or added.

## ⚙️ Installation and Setup

### Prerequisites

- Python version compatible with the project's dependencies
- Git
- Jupyter Notebook support

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Replace the placeholders with your actual GitHub repository URL and directory name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If using `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

If using the project's existing `uv` configuration:

```bash
uv sync
```

Keep the dependency files consistent with the environment you use to run the notebook.

### 4. Open the notebook

From the repository root:

```bash
jupyter notebook mlmodel/student.ipynb
```

Run the cells in order, beginning with data loading and preparation and continuing through model training, evaluation, tuning, and inference.

The notebook expects the CSV file to be available at the path specified in its data-loading code.

## 🔁 Reproducibility

To make experiments easier to reproduce:

- Keep the train-test split random seed consistent.
- Record Python and package versions.
- Preserve the hyperparameter search space and evaluation settings.
- Run the notebook from a clean kernel to verify cell order and dependencies.
- Keep preprocessing and categorical encoding consistent between training and inference.
- Re-evaluate the selected model using a fresh test set when conducting a final unbiased comparison.

The current test set has already been used during model experimentation. Therefore, the reported test metrics should be treated as results from the current experiment rather than a completely untouched final benchmark.

## ⚠️ Limitations and Considerations

- Predictions depend on the quality and representativeness of the dataset.
- High scores on one dataset split do not guarantee performance on different student populations.
- Individual predictions may have substantial errors even when aggregate metrics are strong.
- The model identifies statistical relationships; it does not establish causal relationships between study habits and performance.
- Categorical encoding must be consistent during training and inference.
- The saved `.pkl` file should only be loaded from a trusted source because pickle-based serialization can execute code during deserialization.
- The current implementation is notebook-based. A production API, web interface, deployment pipeline, and monitoring system have not been included in this project description.

## 🚀 Future Improvements

- Create a reusable preprocessing and model pipeline.
- Add automated input validation and data-quality checks.
- Add unit tests for preprocessing, loading, and prediction.
- Compare candidate models using consistent cross-validation.
- Track experiments and model versions.
- Build a FastAPI endpoint for prediction.
- Add a user interface for entering student features.
- Introduce automated testing and deployment workflows.
- Monitor model performance and data drift if suitable real-world data becomes available.

## 📄 License

No license has been specified for this project yet. Add a `LICENSE` file if you intend to publish the source code under an open-source license.

Check the dataset's license separately. A license for your code does not automatically grant permission to redistribute the dataset.

