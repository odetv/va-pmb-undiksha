#!/bin/bash

# Variabel direktori proyek
PROJECT_DIR=~/project/chatbot-pmb-undiksha
VENV_DIR=$PROJECT_DIR/venv

# Cek apakah direktori proyek ada
if [ ! -d "$PROJECT_DIR" ]; then
  echo "Direktori proyek tidak ditemukan. Membuat direktori dan meng-clone repositori..."
  mkdir -p $PROJECT_DIR
  cd $PROJECT_DIR
  git clone https://github.com/username/repository.git .
else
  echo "Direktori proyek ditemukan. Memasuki direktori..."
  cd $PROJECT_DIR || { echo "Gagal masuk ke direktori proyek!"; exit 1; }
fi

# Tarik pembaruan terbaru dari repositori Git
git pull origin main

# Cek apakah virtual environment ada, jika tidak, buat yang baru
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment tidak ditemukan. Membuat virtual environment..."
  python3 -m venv $VENV_DIR
fi

# Aktifkan virtual environment
source $VENV_DIR/bin/activate

# Hentikan aplikasi FastAPI yang sedang berjalan
PIDS=$(ps aux | grep "uvicorn main:app --port=1014" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

# Install atau update dependencies
pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasi FastAPI menggunakan uvicorn
nohup uvicorn main:app --host 0.0.0.0 --port 1014 > output.log 2>&1 &

echo "Deploy selesai. Aplikasi FastAPI sudah berjalan."
