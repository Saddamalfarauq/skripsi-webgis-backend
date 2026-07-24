FROM python:3.10-slim

WORKDIR /code

# Instalasi dependensi sistem untuk GDAL, Geopandas, dan OpenCV (jika dibutuhkan)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Konfigurasi GDAL (diperlukan untuk Rasterio)
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

COPY ./requirements.txt /code/requirements.txt

# Install pip requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy seluruh folder backend
COPY . /code

# Hugging Face menggunakan port 7860
EXPOSE 7860

# Jalankan FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
