from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from database import Base
import datetime

class PredictionRecord(Base):
    # 1. Name the table in SQL
    __tablename__ = "predictions"

    # 2. Define Columns
    id = Column(Integer, primary_key=True, index=True) # Unique ID (1, 2, 3...)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow) # When did it happen?
    
    # 3. The Inputs (We save the 20 raw numbers as a JSON list)
    input_features = Column(JSON)
    
    # 4. The Outputs
    anomaly_score = Column(Float)
    threshold = Column(Float)
    status = Column(String) # "HEALTHY" or "CRITICAL"