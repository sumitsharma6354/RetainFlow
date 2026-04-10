from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd
from datetime import datetime, timedelta
import os
import json

from database.db import engine, init_db, get_db, SessionLocal, Customer, Prediction, RetentionAction
from src.predict import predict_customer, load_models
from src.retention import get_retention_action

app = FastAPI(title="RetainFlow")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def startup_event():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Customer).count() == 0:
            seed_database(db)
    except Exception as e:
        print(f"Critical Error during startup seeding: {str(e)}")
        import traceback
        traceback.print_exc()
        print("💡 Suggestion: Run 'python seed_db.py' manually to fix the database.")
    finally:
        db.close()

def seed_database(db: Session):
    try:
        print("Starting automated database seeding...")
        raw_path = "data/raw/telco_churn.csv"
        if not os.path.exists(raw_path):
            print("❌ Dataset not found at data/raw/telco_churn.csv. Skipping seeding.")
            return

        df = pd.read_csv(raw_path)
        total_rows = len(df)
        
        # Load models
        load_models()
        
        customers = []
        predictions = []
        retention_actions = []
        
        for i, row in df.iterrows():
            customer_id = row['customerID']
            
            # Create Customer record
            customer = Customer(
                customer_id=customer_id,
                gender=row['gender'],
                senior_citizen=int(row['SeniorCitizen']),
                partner=row['Partner'],
                dependents=row['Dependents'],
                tenure=int(row['tenure']),
                phone_service=row['PhoneService'],
                multiple_lines=row['MultipleLines'],
                internet_service=row['InternetService'],
                online_security=row['OnlineSecurity'],
                online_backup=row['OnlineBackup'],
                device_protection=row['DeviceProtection'],
                tech_support=row['TechSupport'],
                streaming_tv=row['StreamingTV'],
                streaming_movies=row['StreamingMovies'],
                contract=row['Contract'],
                paperless_billing=row['PaperlessBilling'],
                payment_method=row['PaymentMethod'],
                monthly_charges=float(row['MonthlyCharges']),
                total_charges=pd.to_numeric(row['TotalCharges'], errors='coerce') if not pd.isna(row['TotalCharges']) else 0.0,
                churn_actual=1 if row['Churn'] == 'Yes' else 0
            )
            customers.append(customer)
            
            # Predictions (Disable SHAP for seeding to avoid overhead)
            feat_dict = row.to_dict()
            feat_dict['customerID'] = customer_id
            res = predict_customer(feat_dict, include_shap=False)
            
            pred = Prediction(
                customer_id=customer_id,
                churn_probability=res['churn_probability'],
                churn_prediction=res['churn_prediction'],
                risk_level=res['risk_level'],
                segment=res['segment'],
                predicted_at=datetime.utcnow()
            )
            predictions.append(pred)
            
            # Retention
            ret_res = get_retention_action(feat_dict, res['churn_probability'], res['segment'])
            ret = RetentionAction(
                customer_id=customer_id,
                offer=ret_res['offer'],
                urgency=ret_res['urgency'],
                status="to_contact",
                estimated_save_value=ret_res['estimated_save_value'],
                created_at=datetime.utcnow()
            )
            retention_actions.append(ret)
            
            if len(customers) >= 500: # Batch insert
                db.bulk_save_objects(customers)
                db.bulk_save_objects(predictions)
                db.bulk_save_objects(retention_actions)
                db.commit()
                customers, predictions, retention_actions = [], [], []
                print(f"Automated Seeding: {i + 1}/{total_rows}...")

        if customers:
            db.bulk_save_objects(customers)
            db.bulk_save_objects(predictions)
            db.bulk_save_objects(retention_actions)
            db.commit()
        
        print("✨ Database seeded successfully.")
    except Exception as e:
        db.rollback()
        raise e

# --- UI ROUTES ---

@app.get("/")
def read_root():
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/customers", response_class=HTMLResponse)
def customers_page(request: Request):
    return templates.TemplateResponse("customers.html", {"request": request})

@app.get("/customer/{customer_id}", response_class=HTMLResponse)
def customer_detail(request: Request, customer_id: str, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return templates.TemplateResponse("customer_detail.html", {"request": request, "customer_id": customer_id})

@app.get("/segments", response_class=HTMLResponse)
def segments_page(request: Request):
    return templates.TemplateResponse("segments.html", {"request": request})

@app.get("/retention", response_class=HTMLResponse)
def retention_page(request: Request):
    return templates.TemplateResponse("retention.html", {"request": request})

# --- API ROUTES ---

@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(Customer).count()
    churn_count = db.query(Prediction).filter(Prediction.churn_prediction == True).count()
    high_risk = db.query(Prediction).filter(Prediction.risk_level == "High").count()
    medium_risk = db.query(Prediction).filter(Prediction.risk_level == "Medium").count()
    low_risk = db.query(Prediction).filter(Prediction.risk_level == "Low").count()
    
    # Revenue at risk: Sum MonthlyCharges of those with high/medium risk
    revenue_at_risk = db.query(func.sum(Customer.monthly_charges))\
        .join(Prediction, Customer.customer_id == Prediction.customer_id)\
        .filter(Prediction.risk_level.in_(["High", "Medium"])).scalar() or 0
    
    retained = db.query(RetentionAction).filter(RetentionAction.status == "retained").count()
    
    # Simple simulated trend
    trend = [
        {"month": "Oct", "rate": 24.2},
        {"month": "Nov", "rate": 25.1},
        {"month": "Dec", "rate": 23.8},
        {"month": "Jan", "rate": 26.4},
        {"month": "Feb", "rate": 25.9},
        {"month": "Mar", "rate": round((churn_count/total)*100, 1) if total > 0 else 0}
    ]
    
    return {
        "total_customers": total,
        "churn_rate": round((churn_count/total)*100, 1) if total > 0 else 0,
        "high_risk_count": high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count": low_risk,
        "monthly_revenue_at_risk": round(float(revenue_at_risk), 2),
        "retained_this_month": retained,
        "churn_trend": trend
    }

@app.get("/api/customers")
def get_customers(
    page: int = 1, 
    limit: int = 10, 
    risk: str = "all", 
    contract: str = "all", 
    search: str = "", 
    db: Session = Depends(get_db)
):
    query = db.query(Customer).join(Prediction, Customer.customer_id == Prediction.customer_id)
    
    if risk != "all":
        query = query.filter(Prediction.risk_level == risk)
    if contract != "all":
        query = query.filter(Customer.contract == contract)
    if search:
        query = query.filter(Customer.customer_id.contains(search))
        
    total = query.count()
    customers = query.offset((page-1)*limit).limit(limit).all()
    
    res = []
    for c in customers:
        res.append({
            "customer_id": c.customer_id,
            "tenure": c.tenure,
            "monthly_charges": c.monthly_charges,
            "contract": c.contract,
            "risk_level": c.prediction.risk_level,
            "churn_probability": c.prediction.churn_probability,
            "segment": c.prediction.segment
        })
        
    return {"total": total, "customers": res}

@app.get("/api/customer/{id}")
def get_customer_api(id: str, db: Session = Depends(get_db)):
    c = db.query(Customer).filter(Customer.customer_id == id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Not found")
    
    # Re-run prediction to get SHAP values (not stored in DB for space)
    feat_dict = {
        "gender": c.gender,
        "SeniorCitizen": c.senior_citizen,
        "Partner": c.partner,
        "Dependents": c.dependents,
        "tenure": c.tenure,
        "PhoneService": c.phone_service,
        "MultipleLines": c.multiple_lines,
        "InternetService": c.internet_service,
        "OnlineSecurity": c.online_security,
        "OnlineBackup": c.online_backup,
        "DeviceProtection": c.device_protection,
        "TechSupport": c.tech_support,
        "StreamingTV": c.streaming_tv,
        "StreamingMovies": c.streaming_movies,
        "Contract": c.contract,
        "PaperlessBilling": c.paperless_billing,
        "PaymentMethod": c.payment_method,
        "MonthlyCharges": c.monthly_charges,
        "TotalCharges": c.total_charges
    }
    
    pred_res = predict_customer(feat_dict)
    ret_res = get_retention_action(feat_dict, pred_res['churn_probability'], pred_res['segment'])
    
    return {
        "customer": {
            "customer_id": c.customer_id,
            "gender": c.gender,
            "senior_citizen": c.senior_citizen,
            "partner": c.partner,
            "dependents": c.dependents,
            "tenure": c.tenure,
            "phone_service": c.phone_service,
            "multiple_lines": c.multiple_lines,
            "internet_service": c.internet_service,
            "online_security": c.online_security,
            "online_backup": c.online_backup,
            "device_protection": c.device_protection,
            "tech_support": c.tech_support,
            "streaming_tv": c.streaming_tv,
            "streaming_movies": c.streaming_movies,
            "contract": c.contract,
            "paperless_billing": c.paperless_billing,
            "payment_method": c.payment_method,
            "monthly_charges": c.monthly_charges,
            "total_charges": c.total_charges
        },
        "prediction": pred_res,
        "retention": ret_res,
        "retention_status": c.retention_action.status
    }

@app.get("/api/segments")
def get_segments(db: Session = Depends(get_db)):
    segments = db.query(Prediction.segment).distinct().all()
    res = []
    for s_name, in segments:
        count = db.query(Prediction).filter(Prediction.segment == s_name).count()
        avg_prob = db.query(Prediction).filter(Prediction.segment == s_name).with_entities(func.avg(Prediction.churn_probability)).scalar()
        avg_tenure = db.query(Customer).join(Prediction).filter(Prediction.segment == s_name).with_entities(func.avg(Customer.tenure)).scalar()
        avg_charges = db.query(Customer).join(Prediction).filter(Prediction.segment == s_name).with_entities(func.avg(Customer.monthly_charges)).scalar()
        
        # Churn rate per segment
        churn_count = db.query(Prediction).filter(Prediction.segment == s_name, Prediction.churn_prediction == True).count()
        
        res.append({
            "name": s_name,
            "count": count,
            "avg_churn_prob": round(float(avg_prob), 3) if avg_prob else 0,
            "avg_tenure": round(float(avg_tenure), 1) if avg_tenure else 0,
            "avg_monthly_charges": round(float(avg_charges), 2) if avg_charges else 0,
            "churn_rate": round((churn_count/count)*100, 1) if count > 0 else 0
        })
    return res

@app.get("/api/charts/churn-by-contract")
def churn_by_contract(db: Session = Depends(get_db)):
    data = db.query(Customer.contract, func.count(Customer.id))\
        .join(Prediction).filter(Prediction.churn_prediction == True).group_by(Customer.contract).all()
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    return {"labels": labels, "values": values}

@app.get("/api/charts/churn-by-internet")
def churn_by_internet(db: Session = Depends(get_db)):
    data = db.query(Customer.internet_service, func.count(Customer.id))\
        .join(Prediction).filter(Prediction.churn_prediction == True).group_by(Customer.internet_service).all()
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    return {"labels": labels, "values": values}

@app.get("/api/charts/tenure-vs-charges")
def tenure_vs_charges(db: Session = Depends(get_db)):
    # Limit to 500 points for performance
    data = db.query(Customer.tenure, Customer.monthly_charges, Prediction.segment, Prediction.risk_level)\
        .join(Prediction).limit(500).all()
    return [{"x": d[0], "y": d[1], "segment": d[2], "risk": d[3]} for d in data]

@app.post("/api/retention/update")
async def update_retention(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    customer_id = data.get("customer_id")
    status = data.get("status")
    
    ret = db.query(RetentionAction).filter(RetentionAction.customer_id == customer_id).first()
    if ret:
        ret.status = status
        db.commit()
        return {"success": True}
    return {"success": False}

@app.post("/api/predict")
async def api_predict(request: Request):
    data = await request.json()
    return predict_customer(data)
