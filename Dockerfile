# Gunakan Python 3.9 sebagai base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt ke dalam container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project ke dalam container
COPY . /app

# Expose port 1014
EXPOSE 1014

# Command untuk menjalankan aplikasi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1014"]
