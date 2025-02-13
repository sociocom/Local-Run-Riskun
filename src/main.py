import re
import pandas as pd
from datetime import datetime
from run_llm import download_model, generate
import streamlit as st
import streamlit_ext as ste
from io import StringIO
import os

def set_streamlit():
    # カスタムテーマの定義
    st.set_page_config(
        page_title="リスくん - リスク因子構造化システム",
        page_icon=":chipmunk:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.extremelycoolapp.com/help",
            "Report a bug": "https://www.extremelycoolapp.com/bug",
            "About": """
            # リスくん:chipmunk: リスク因子構造化システム
            脳卒中のリスク因子を構造化し、csv形式で出力するシステムです。""",
        },
    )
    st.title("リスくん:chipmunk: リスク因子構造化システム")
    st.markdown("###### 脳卒中のリスク因子を構造化し、csv形式で出力するシステムです。")

    st.sidebar.write("### サンプルファイルで実行する場合は以下のファイルをダウンロードしてください。")
    sample_csv = pd.read_csv("data/sample3.csv")
    sample_csv = sample_csv.to_csv(index=False)
    ste.sidebar.download_button("sample data", sample_csv, "riskun_sample.csv")

    st.sidebar.markdown("### 因子構造化に用いるcsvファイルを選択してください。")
    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", accept_multiple_files=False
    )
    return uploaded_file

def convert_to_utf8(content, encoding):
    try:
        return content.decode(encoding).encode("utf-8")
    except UnicodeDecodeError:
        return None

@st.cache_data
def read_uploaded_file_as_utf8(uploaded_file):
    # ファイルをバイナリモードで読み込み
    content = uploaded_file.read()

    # エンコーディングを自動検出し、UTF-8に変換
    encodings_to_try = [
        "utf-8",
        "shift-jis",
        "cp932",
        "latin-1",
        "ISO-8859-1",
        "euc-jp",
        "euc-kr",
        "big5",
        "utf-16",
    ]
    utf8_content = None

    for encoding in encodings_to_try:
        utf8_content = convert_to_utf8(content, encoding)
        if utf8_content is not None:
            break

    try:
        df = pd.read_csv(StringIO(utf8_content.decode("utf-8")))
    except pd.errors.EmptyDataError:
        st.error("データが読み込めませんでした。utf-8のエンコードのcsvファイルを選んでください。")

    return df


os.environ["STREAMLIT_CONFIG"] = "config.toml"

# def main():
uploaded_file = set_streamlit()

if uploaded_file:
    df = read_uploaded_file_as_utf8(uploaded_file)
    st.write(f"入力ファイル全{len(df)}件 (先頭5件までを表示)")
    st.dataframe(df.head(5))
    target_columns = st.selectbox(
        "カルテが記載されている項目を選んでください",
        (df.columns),
        index=None,
        placeholder="Select...",
    )
    st.write("選択した項目:", target_columns)

    if target_columns:
        with st.spinner("実行中..."):
            tokenizer, model = download_model(model_name="elyza/Llama-3-ELYZA-JP-8B")
            for i in range(len(df)):
                df_json = generate(target_columns, df[target_columns][i], tokenizer, model)
                display_df = df_json.copy()
                display_df[target_columns] = display_df[target_columns].str[:10] + "..."
                display_df["診断名"] = display_df["診断名"].apply(
                    lambda x: x[:5] + "..." if isinstance(x, str) and len(x) > 5 else x
                )
                display_df["プロブレムリスト"] = display_df["プロブレムリスト"].apply(
                    lambda x: x[:5] + "..." if isinstance(x, str) and len(x) > 5 else x
                )
                display_df["外科治療歴の有無"] = display_df["外科治療歴の有無"].apply(
                    lambda x: x[:5] + "..." if isinstance(x, str) and len(x) > 5 else x
                )
                if i == 0:
                    output_df = df_json
                    mytable = st.table(display_df)
                else:
                    output_df = pd.concat([output_df, df_json],axis=0)
                    mytable.add_rows(display_df)

        st.write("出力結果 (先頭5件までを表示)")
        st.dataframe(output_df.head(5))
        if "completed" not in st.session_state:
            st.session_state["completed"] = True

        file_name = uploaded_file.name.replace(".csv", "").replace("riskun_", "")
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        csv = output_df.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode("utf-8-sig")).decode()
        # href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}-riskun-{timestamp}.csv">Download Link</a>'
        # st.markdown(f"CSVファイルのダウンロード: {href}", unsafe_allow_html=True)

        if ste.download_button(
            "Click to download data", csv, f"{file_name}-riskun-{timestamp}.csv"
        ):
            st.cache_data.clear()
            st.cache_resource.clear()
            # セマフォ解放（必要に応じて）
            os.system("ipcs -s | awk 'NR>2 {print $2}' | xargs -n 1 ipcrm -s")

        # # 修正後のcsvアップロード
        # fixed_file = st.sidebar.file_uploader(
        # "修正したcsvファイルがあればアップロードしてください。", accept_multiple_files=False
        # )
        # if fixed_file:
        #     fixed_file.to_csv(f"../data/fixed/{file_name}-riskun-{timestamp}.csv")

        # クライアント側のJavaScriptコードを埋め込み
        shutdown_script = """
        <script>
            // タブが閉じられたときにサーバーにリクエストを送信
            window.addEventListener('beforeunload', function (event) {
                navigator.sendBeacon('/shutdown');
            });
        </script>
        """
        st.markdown(shutdown_script, unsafe_allow_html=True)