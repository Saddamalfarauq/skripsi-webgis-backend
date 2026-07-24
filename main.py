from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routes import predict

# Membuat tabel database jika belum ada
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Flood Detection WebGIS API",
    description="Backend API untuk memprediksi potensi banjir dari citra satelit",
    version="1.0.0"
)

# Konfigurasi CORS agar bisa diakses oleh Frontend (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ganti dengan domain React di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan routes
app.include_router(predict.router, prefix="/api", tags=["Predict"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Flood Detection API. Gunakan /api/predict untuk inferensi."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
