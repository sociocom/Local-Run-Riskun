#!/bin/bash
cd /Users/sociocom/research/riskun

source .venv/bin/activate

# ポート 8501 を占有しているプロセスを終了
if lsof -t -i:8501 > /dev/null; then
  echo "ポート8501を解放中..."
  lsof -t -i:8501 | xargs kill
  sleep 2  # プロセス終了を待機
fi

streamlit run src/main.py --server.port=8501