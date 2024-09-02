# Gunakan image Python sebagai base
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Copy semua file yang dibutuhkan untuk aplikasi
COPY . .

# Install dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port untuk FastAPI
EXPOSE 1014

# Perintah untuk menjalankan aplikasi menggunakan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1014"]
