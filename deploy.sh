PROJECT_DIR=~/home/bot/project/chatbot-pmb-undiksha

cd $PROJECT_DIR

git pull origin main

echo "Sedang berada di direktori: $(pwd)"

PIDS=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "Menghentikan aplikasi dengan PID: $PIDS"
  kill -9 $PIDS
else
  echo "Tidak ada aplikasi yang berjalan."
fi

nohup ~/venv/bin/python main.py > output.log 2>&1 &

echo "Deploy selesai. Aplikasi FastAPI sudah berjalan."