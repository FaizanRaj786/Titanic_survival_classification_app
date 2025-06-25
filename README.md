# 🚢 Titanic Survival Predictor

A complete machine learning pipeline that predicts survival chances of Titanic passengers.  
This project includes data preprocessing, model training with hyperparameter tuning, and a **Streamlit-based web app** for live predictions.

---

## 📌 Project Features

- Full ML pipeline using `scikit-learn`
- Feature engineering on Titanic dataset
- Hyperparameter tuning via `GridSearchCV`
- Final model: `RandomForestClassifier` (optimized)
- GUI made with **Streamlit** for live predictions

---

## 🧠 Dataset

Used [Kaggle's Titanic dataset](https://www.kaggle.com/competitions/titanic).  
Target column: `Survived` (0 = Died, 1 = Survived)

---

## 🧪 ML Workflow

1. **Data Collection**
2. **Cleaning & Imputation**
3. **Feature Engineering**
   - Title extraction from Name
   - AgeBand & FareBand binning
   - Family size & IsAlone features
4. **Train-Validation-Test split**
5. **Model Training**  
   Models tested:
   - Logistic Regression
   - Decision Tree
   - Random Forest ✅ (Best)
   - XGBoost
6. **Hyperparameter Tuning** with `GridSearchCV`
7. **Evaluation** using accuracy & confusion matrix
8. **Deployment** using `Streamlit`

---

## 🎯 Final Model Performance

- **Best Model:** Random Forest (tuned)
- **Validation Accuracy:** ~79.1%
- **Cross-Validation Accuracy:** ~82.3%
- **Test Accuracy:** ~[insert after test evaluation]

---

## 🖥️ How to Run the Streamlit App

### 🔧 1. Install dependencies

```bash
pip install -r requirements.txt
