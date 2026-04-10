# RetainFlow

**RetainFlow** (formerly ChurnGuard) is an AI-powered customer retention and churn prediction platform. By analyzing telecom customer data, RetainFlow predicts which users are at risk of leaving, segments them by risk level, and suggests actionable retention strategies.

## 🚀 Features

- **Churn Prediction Dashboard**: Monitor overall churn rate, revenue at risk, and recent retention actions in a real-time dashboard.
- **Machine Learning Integration**: Built with `XGBoost` and `scikit-learn` to reliably train on data and infer churn probabilities. Uses `SHAP` to understand feature importance.
- **Risk Segmentation**: Categorizes customers into "Low", "Medium", and "High" risk levels, allocating specific retention actions based on risk and customer value.
- **Interactive UI**: A sleek, user-friendly frontend to quickly search customers by ID, contract type, or risk level.
- **RESTful API endpoints**: Fully accessible documented endpoints for retrieving statistics, segment data, and trigger model predictions.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python, SQLAlchemy
- **Machine Learning**: XGBoost, Scikit-Learn, Pandas, SHAP
- **Database**: SQLite
- **Frontend**: HTML, CSS, JavaScript, Jinja2 Templates, Chart.js

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/retainflow.git
   cd retainflow
   ```

2. **Install the dependencies:**
   Make sure you have Python installed. You can install all dependencies via `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Seed Database and Train Model:**
   You can seed the database with initial customer records.
   ```bash
   python seed_db.py
   ```
   To train the machine learning pipeline from scratch:
   ```bash
   python train_pipeline.py
   ```

4. **Run the application:**
   Launch the FastAPI web server.
   ```bash
   uvicorn main:app --reload
   ```

5. **Open in Browser:**
   Navigate to `http://localhost:8000` to interact with your local instance of RetainFlow.

## ✨ Project Structure

- `main.py` - FastAPI entrypoint and route definitions.
- `database/` - SQLAlchemy models, DB setup, and configuration.
- `src/` - Core logic for model training (`train.py`), prediction handling (`predict.py`), and determining retention actions (`retention.py`).
- `data/` - Holds raw CSV inputs (like `telco_churn.csv`).
- `templates/` & `static/` - UI rendering components.
