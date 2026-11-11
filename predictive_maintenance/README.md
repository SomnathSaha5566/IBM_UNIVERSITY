# Predictive Maintenance of Industrial Machinery (PS 39)

IBM Skills Build / AICTE-2026 - ML project. A multi-class classifier that predicts the **type of machine
failure** (Heat Dissipation, Power, Overstrain, Tool Wear, Random, or No Failure) from sensor readings,
with a Streamlit dashboard and deployment to **IBM Watson Machine Learning (Cloud Lite)**.

## Project layout
```
app.py                       Streamlit dashboard
src/features.py              cleaning + physics-informed feature engineering
src/train.py                 model comparison (CV), evaluation, plots, saves model
src/predict.py               CLI prediction
src/ibm_deploy.py            deploy to IBM Watson Machine Learning
src/generate_synthetic_data.py   fake data with same schema (smoke tests only)
data/                        put the dataset CSV here
models/  reports/            outputs (model.joblib, confusion matrix, metrics.json)
```

## Run it
```bash
pip install -r requirements.txt

# 1. Get the real dataset (needs a Kaggle login):
#    https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
#    Save it as data/predictive_maintenance.csv
#    (no dataset yet? `python src/generate_synthetic_data.py` creates a stand-in.
#     data/predictive_maintenance.csv currently holds SYNTHETIC data - replace it!)

python src/train.py                       # trains, evaluates, writes models/ and reports/
python src/predict.py --type L --air 298.5 --process 308.6 --rpm 1400 --torque 45 --wear 210
streamlit run app.py                      # dashboard
```

## Approach
1. **Cleaning** - drops rows where `Target` and `Failure Type` contradict each other (a known issue in this dataset).
2. **Feature engineering** - temperature difference (heat dissipation), power = torque x angular speed
   (power failure), tool wear x torque (overstrain). These mirror the physical failure modes.
3. **Imbalance** - failures are ~3% of rows, so models use `class_weight="balanced"` and are judged on
   **macro-F1**, not accuracy. Classes with <10 samples (usually "Random Failures") are dropped.
4. **Model selection** - Logistic Regression vs Random Forest vs HistGradientBoosting, 5-fold stratified CV;
   best one is refit and evaluated on a held-out 20% test set.
5. **Outputs** - `reports/confusion_matrix.png`, `reports/feature_importance.png`, `reports/metrics.json`.

## IBM Cloud Lite deployment
1. Create a free IBM Cloud Lite account; provision **Watson Machine Learning** (Lite) and **Watson Studio**.
2. Create a project + **deployment space**; copy its Space ID. Create an IAM API key.
3. ```bash
   export IBM_API_KEY=...  IBM_SPACE_ID=...  IBM_URL=https://us-south.ml.cloud.ibm.com
   python src/ibm_deploy.py
   ```
   WML runtimes pin scikit-learn versions (the script targets `scikit-learn_1.3`). If the upload reports a
   version mismatch, run `pip install scikit-learn==1.3.2`, retrain, and redeploy.
4. Tip for the submission: also run `src/train.py` inside a Watson Studio notebook and add screenshots of
   the deployment space to show IBM services are used.

## Possible extensions
- SMOTE oversampling, hyperparameter search, SHAP explanations
- Granite-powered assistant that explains a prediction in plain language (`ibm-watsonx-ai` ModelInference)
