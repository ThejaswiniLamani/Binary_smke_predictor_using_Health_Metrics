# Binary smoke predictor using Bio-Signals and Health Metrics

This project uses multiple machine learning algorithms to predict whether an individual smokes or not based on various health-related features. The models are evaluated using performance metrics like accuracy, AUC-ROC, and cross-validation scores.

---

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Bio-Signals](#bio-signals)
- [Health Parameters](#health-parameters)
- [Algorithms Used](#algorithms-used)
- [Model Evaluation](#model-evaluation)
- [Files](#files)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Project Description
The goal of this project is to develop a machine learning model that predicts smoking behavior based on health and demographic data. The dataset contains various health-related features such as height, weight, and hemoglobin levels.

The project compares the performance of multiple machine learning algorithms including Logistic Regression, Decision Tree, Random Forest, and XGBoost to assess which model provides the best prediction accuracy.

---

## Dataset
The dataset used in this project consists of various health-related features that can be used to predict smoking behavior. The dataset has two primary classes:

- **Class 0**: Non-Smoking
- **Class 1**: Smoking

### Bio-Signals
Bio-signals are physiological readings or markers that reflect the state of the body. The following features are included:
- **Eyesight (left/right)**: Visual ability measures.
- **Hearing (left/right)**: Hearing ability measures.
- **Hemoglobin**: Oxygen-carrying protein in red blood cells.
- **Serum Creatinine**: Marker for kidney function.
- **AST (Aspartate Aminotransferase)**: Liver enzyme.
- **ALT (Alanine Aminotransferase)**: Liver enzyme.
- **GTP (Gamma-glutamyl Transferase)**: Enzyme linked with liver health.

### Health Parameters
Health parameters are broader metrics relating to overall health and risk factors:
- **Age**: Demographic data.
- **Height (cm)** and **Weight (kg)**: General health indicators.
- **Waist (cm)**: Body fat distribution.
- **Systolic**: Blood pressure reading.
- **Relaxation**: Stress or relaxation level.
- **Fasting Blood Sugar**: Indicator for diabetes risk.
- **Cholesterol**: Heart health parameter.
- **Triglyceride**: Blood fat level.
- **HDL/LDL**: Good and bad cholesterol levels.
- **Urine Protein**: Marker for kidney damage.
- **Dental Caries**: Oral health indicator.

---

## Algorithms Used
### Logistic Regression
- **Accuracy**: 0.75
- **AUC-ROC**: 0.83
- **Cross-validation Accuracy**: 0.75

### Decision Tree Classifier
- **Accuracy**: 0.75
- **AUC-ROC**: 0.83
- **Cross-validation Accuracy**: 0.75
- **Model Hyperparameters**: `max_depth=6`

### Random Forest Classifier
- **Accuracy**: 0.76
- **Cross-validation Accuracy**: 0.76
-  **AUC-ROC**: 0.84
- **Model Hyperparameters**: `max_depth=6`, `random_state=42`

### XGBoost Classifier
- **Accuracy**: 0.78
- **Cross-validation Accuracy**: 0.78
-  **AUC-ROC**: 0.86
- **Model Hyperparameters**:
  - `max_depth=10`
  - `learning_rate=0.05`
  - `n_estimators=150`
  - `eval_metric='logloss'`
  - `random_state=42`
  - 
 ### Gradient Boosting
- **Accuracy**: 0.77
- **Cross-validation Accuracy**: 0.77


---

## Model Evaluation
### Metrics Used
- **Accuracy**: Proportion of correctly classified instances.
- **AUC-ROC**: Area Under the Curve of the Receiver Operating Characteristic.
- **Cross-validation Accuracy**: 10-fold cross-validation to assess model generalization.

---

## Files
- **`smoke.ipynb`**: Code for data preprocessing, model training, and evaluation.
- **`smoke_dataset.csv`**: Main dataset used for training and testing.
- **`smoke_dataset-test.csv`**: Unseen data used for predictions.
- **`smoke_dataset_test_with_prediction.csv`**: Predictions on test data with smoker status.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nirajccs1999/Predicting-Smoker-Status-Using-Bio-Signals-and-Health-Metrics.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Predicting-Smoker-Status-Using-Bio-Signals-and-Health-Metrics
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook smoke.ipynb
   ```
2. Follow the instructions in the notebook to preprocess data, train models, and evaluate performance.

---

## Results
### Logistic Regression
- **Accuracy**: 0.75
- **AUC-ROC**: 0.83
- **Cross-validation Accuracy**: 0.75

### Decision Tree Classifier
- **Accuracy**: 0.75
- **AUC-ROC**: 0.83
- **Cross-validation Accuracy**: 0.75

### Random Forest Classifier
- **Accuracy**: 0.76
- **Cross-validation Accuracy**: 0.76
- - **AUC-ROC**: 0.84


### XGBoost Classifier
- **Accuracy**: 0.78
- **Cross-validation Accuracy**: 0.78
- - **AUC-ROC**: 0.86


### Gradient Boosting
- **Accuracy**: 0.77
- **Cross-validation Accuracy**: 0.77



---

## Conclusion
XGBoost emerged as the top-performing model with the highest accuracy (77%) and cross-validation score (~78%). Logistic Regression and Decision Tree followed, both achieving 75% accuracy, while Random Forest performed slightly better with 76% accuracy. This demonstrates the efficacy of ensemble methods for predicting smoker status.
