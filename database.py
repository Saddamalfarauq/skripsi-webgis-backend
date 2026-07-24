import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Menggunakan PostgreSQL secara default, tapi Anda dapat mengubah kredensialnya di sini
# Format: postgresql://[user]:[password]@[host]:[port]/[db_name]
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:parkshijin743@localhost:5432/flood_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
