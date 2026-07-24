from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    location_name = Column(String, index=True, default="Unknown")
    risk_level = Column(String, nullable=False)
    risk_confidence = Column(Float, nullable=False)
    flooded_regions_count = Column(Integer, default=0)
    geojson_path = Column(String, nullable=True) # Path/URL ke file GeoJSON
    image_path = Column(String, nullable=True)   # Path/URL ke visualisasi asli
    created_at = Column(DateTime, default=datetime.utcnow)
