#!/bin/bash

# Direktori proyek
PROJECT_DIR=~/project/chatbot-pmb-undiksha
# Direktori virtual environment
VENV_DIR=$PROJECT_DIR/venv

# Buat direktori proyek jika belum ada
mkdir -p $PROJECT_DIR

# Pindah ke direktori proyek
cd $PROJECT_DIR || exit

# Tarik perubahan terbaru dari GitHub
git pull origin main

echo "Sedang berada di direktori: $(pwd)"

# Membuat virtual environment jika belum ada
if [ ! -d "$VENV_DIR" ]; then
  echo "Membuat virtual environment di $VENV_DIR"
  python3 -m venv venv
fi

# Mengaktifkan virtual environment
echo "Mengaktifkan virtual environment"
source $VENV_DIR/bin/activate

# Menginstal dependensi dari requirements.txt
echo "Menginstal dependensi dari requirements.txt"
pip install --no-cache-dir -r requirements.txt

# Hentikan proses aplikasi yang sedang berjalan
PIDS=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

# Jalankan aplikasi menggunakan nohup
echo "Menjalankan aplikasi..."
nohup python main.py > output.log 2>&1 &

echo "Deploy selesai. Aplikasi FastAPI sudah berjalan."
