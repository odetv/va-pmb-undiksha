#!/bin/bash

# Variabel direktori proyek
PROJECT_DIR=~/project/chatbot-pmb-undiksha
VENV_DIR=$PROJECT_DIR/venv

# Cek apakah direktori proyek ada
if [ ! -d "$PROJECT_DIR" ]; then
  echo "Direktori proyek tidak ditemukan. Membuat direktori dan meng-clone repositori..."
  mkdir -p $PROJECT_DIR
  cd $PROJECT_DIR
  git clone https://github.com/odetv/chatbot-pmb-undiksha.git .
else
  echo "Direktori proyek ditemukan. Memasuki direktori..."
  cd $PROJECT_DIR || { echo "Gagal masuk ke direktori proyek!"; exit 1; }
fi

# Cek apakah direktori Git ada
if [ ! -d ".git" ]; then
  echo "Direktori bukan git repository. Melakukan git clone..."
  git init
  git remote add origin https://github.com/odetv/chatbot-pmb-undiksha.git
  git pull origin main
else
  echo "Git repository ditemukan. Menarik pembaruan terbaru..."
  git pull origin main
fi

# Cek apakah virtual environment ada, jika tidak, buat yang baru
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment tidak ditemukan. Membuat virtual environment..."
  python3 -m venv $VENV_DIR
fi

# Aktifkan virtual environment
source $VENV_DIR/bin/activate

# Install dependencies dari requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Meninstall dependencies dari requirements.txt..."
  pip install -r requirements.txt
else
  echo "File requirements.txt tidak ditemukan!"
  exit 1
fi

# Membuat file .env dengan variabel dari secrets
echo "Membuat file .env dengan variabel dari secrets..."
cat <<EOL > .env
OPENAI_API_KEY=${OPENAI_API_KEY}
CHATBOT_API_KEY=${CHATBOT_API_KEY}
EOL

# Hentikan aplikasi FastAPI yang sedang berjalan
PIDS=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

# Jalankan aplikasi FastAPI menggunakan uvicorn
sudo nohup $VENV_DIR/bin/python main.py > output.log 2>&1 &

echo "Deploy selesai. Aplikasi FastAPI sudah berjalan."
