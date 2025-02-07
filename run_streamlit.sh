#!/bin/bash
cd /Users/sociocom/research/riskun

source .venv/bin/activate

# ポート 8501 を占有しているプロセスを終了
if lsof -t -i:8501 > /dev/null; then
  echo "ポート8501を解放中..."
  lsof -t -i:8501 | xargs kill
  sleep 2  # プロセス終了を待機
fi

ENV STREAMLIT_THEME_BASE="light"
ENV STREAMLIT_THEME_PRIMARY_COLOR="#FF715B"
ENV STREAMLIT_THEME_BACKGROUND_COLOR="#FFFFFF"
ENV STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#34C3B5"
ENV STREAMLIT_THEME_TEXT_COLOR="#4C5454"
ENV STREAMLIT_THEME_FONT="sans serif"
ENV STREAMLIT_THEME_SIDEBAR_BACKGROUND_COLOR="#FF715B"
ENV STREAMLIT_THEME_SIDEBAR_CONTRAST=1.2

streamlit run src/main.py --server.port=8501