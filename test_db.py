from database.db import SessionLocal, Customer, Prediction
db = SessionLocal()
print(f"Customer count: {db.query(Customer).count()}")
print(f"Prediction count: {db.query(Prediction).count()}")
db.close()
