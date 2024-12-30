# Diabetes Prediction Using Machine Learning

This project predicts whether a person is diabetic or non-diabetic based on various health attributes using the **PIMA Diabetes Dataset** and a Support Vector Machine (SVM) model.

---

## Overview

The prediction system classifies individuals based on health metrics like glucose levels, BMI, and insulin levels. It demonstrates:
- Data preprocessing and standardization
- Model training and evaluation
- Predictive system for new input data

---

## Dataset

The **PIMA Diabetes Dataset** contains 768 samples with 8 features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body Mass Index
- **DiabetesPedigreeFunction**: Family history score
- **Age**: Age in years
- **Outcome**: 0 (Non-diabetic), 1 (Diabetic)

---

## Dependencies

Required Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`

---

## Steps

### 1. Data Preprocessing
- Load and explore the dataset.

### 2. Train-Test Split
- Split data into 80% training and 20% testing subsets.

### 3. Model Training
- Train an SVM classifier with a linear kernel

### 4. Model Evaluation
- Training Accuracy: **78.3%**
- Testing Accuracy: **77.9%**

### 5. Prediction System
- Classify new inputs as diabetic or non-diabetic.

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/iDharitri-Raj/Diabetes_Prediction
```
2. Install dependencies:
```bash
pip install numpy pandas scikit-learn
```
3. Run the script to train the model and make predictions.

---

## Acknowledgments

- **Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)
- **Libraries**: NumPy, Pandas, Scikit-learn
