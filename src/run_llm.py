import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import jaconv
import logging

DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT = """あなたは誠実で優秀な日本人の医師です。以下に示す電子カルテの記事から、指定された項目について情報を抽出し、厳密なJSON形式で出力してください。

#### 抽出する項目:
1. 性別: 男性または女性を記録。記載がない場合は"U"。
2. 身長 (cm): 数値を記載。単位は不要。記載がない場合は"U"。
3. 体重 (kg): 数値を記載。単位は不要。記載がない場合は"U"。
4. 年齢 (歳): 数値を記載。単位は不要。記載がない場合は"U"。
5. HbA1c (%): 数値を記載。単位は不要。記載がない場合は"U"。
6. CRP (mg/dL): 数値を記載。単位は不要。記載がない場合は"U"。
7. 血圧 (mmHg): 収縮期と拡張期の両方を記載（例: "120/80"）。単位は不要。記載がない場合は"U"。
8. 体温 (℃): 数値を記載。単位は不要。記載がない場合は"U"。
9. 脈拍 (回/分): 数値を記載。単位は不要。記載がない場合は"U"。
10. 糖尿病:
    - P: 糖尿病の診断あり
    - N: 糖尿病の診断なし
    - U: 記載なし
11. 喫煙歴:
    - PP: 過去に喫煙歴がある
    - P: 現在も喫煙している
    - N: 喫煙歴なし
    - U: 記載なし
12. 飲酒歴:
    - PP: 過去に飲酒歴がある
    - P: 現在も飲酒している
    - N: 飲酒歴なし
    - U: 記載なし
13. 診断名: 主病名を記載。記載がない場合は"U"。
14. プロブレムリスト: 症状、疑いのある病名、既往歴などを記載。記載がない場合は"U"。
15. 外科治療歴の有無:
    - もやもや病
    - 脳動脈奇形（心房細動、心房粗動、洞不全症候群、B23:K25、その他）
    - 未破裂脳動脈瘤
    - U: 記載なし
16. 発症前mRS:
    - 0: 全く症候がない
    - 1: 症候があるが明らかな障害はない
    - 2: 軽度の障害
    - 3: 中等度の障害（歩行介助なし、何らかの介助必要）
    - 4: 中等度〜重度の障害（介助必要）
    - 5: 重度の障害（寝たきり、失禁状態）
    - U: 記載なし
17. 入院前生活場所:
    - 自宅（独居、家族と同居、サービス付き高齢者施設、詳細不明）
    - 自院
    - リハビリ目的の他病院
    - リハビリ目的以外の他病院
    - その他施設: 介護保険施設、老人ホームなど
    - 介護施設: 老人ホーム、介護保険施設など
    - 不明: 不明と書いてある
    - U: 記載なし
18. （ワルファリン症例）来院時PT-INR: 数値を記載。記載がない場合は"U"。
19. 抗血小板薬に対する使用状況:
	- P: 処方あり
	- N: 処方なし
	- U: 言及無し
	抗血小板薬の薬剤を列挙します。情報抽出にあたり参考にしてください
	アスピリン、バイアスピリン、クロピドグレル、プラビックス、プラスグレル、エフィエント、チクロピジン、バナルジン、トラピジル、ロコルナール、オザグレル、カタクロット、ジピリダモール、ベルサンチン、チカグレロル、ブリリンタ
20. 抗凝固薬に対する使用状況:
	- P: 処方あり
	- N: 処方なし
	- U: 言及無し
	抗凝固薬の薬剤を列挙します。情報抽出にあたり参考にしてください
	ワルファリン、ワーファリン、リバーロキサバン、イグザレルト、アピキサバン、エリキュース、エドキサバン、リクシアナ、ダビガトラン、プラザキサ
21. スタチンに対する使用状況:
	- P: 処方あり
	- N: 処方なし
	- U: 言及無し
	スタチンの薬剤を列挙します。情報抽出にあたり参考にしてください
	アトルバスタチン、リピトール、ロスバスタチン、クレストール、プラバスタチン、メバロチン、シンバスタチン
22. 降圧薬に対する使用状況:
	- P: 処方あり
	- N: 処方なし
	- U: 言及無し
	降圧薬の薬剤を列挙します。情報抽出にあたり参考にしてください
	エナラプリル、リシノプリル、ロサルタン、バルサルタン、アムロジピン、カルベジロール、ニューロタン、ロサルタンK、ロサルタンカリウム、ディオバン、ディオバンOD、バルサルタン、バルサルタンOD、アバプロ、イルベサルタン、イルベタン、ブロプレス、カンデサルタン、カンデサルタンOD、ミカルディス、テルミサルタン、テルミサルタンOD、オルメテック、オルメサルタン、オルメサルタンOD、アジルバ、アジルバ顆粒、アジルサルタン、アジルサルタンOD
23. 糖尿病治療薬に対する使用状況:
	- P: 処方あり
	- N: 処方なし
	- U: 言及無し
	糖尿病治療薬の薬剤を列挙します。情報抽出にあたり参考にしてください
	シタグリプチン、グラクティブ、ビルダグリプチン、エクア、アナグリプチン、スイニー、サキサグリプチン、オングリザ、オマリグリプチン、マリゼブ、アカルボース、グルコバイ1、ピオグリタゾン、アクトス1、ダパグリフロジン、フォシーガ、トホグリフロジン、デベルザ、アプルウェイ、ルセオグリフロジン、ルセフィ1、エキセナチド、バイエッタ、リキシセナチド、リキスミア、セマグルチド、リベルサス1、ピオグリタゾン/メトホルミン、メタクト配合錠、ピオグリタゾン/グリメピリド、ソニアス配合錠、アログリプチン/ピオグリタゾン、リオベル配合錠、ミチグリニド/ボグリボース、グルベス配合錠、テネリグリプチン/カナグリフロジン、カナリア配合錠1

#### 出力形式:
以下のJSON形式で出力してください。JSON形式のルールに厳密に従い、余分な説明やエラーのない正確な出力を行ってください。

```
{
    "性別": "男性",
    "身長": "175",
    ...
    "降圧薬に対する使用状況": "P",
    "糖尿病治療薬に対する使用状況": "U"
}
```
"""


def output_response(DEFAULT_SYSTEM_PROMPT, text, tokenizer, model, temperature=0.1):
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=600,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
    )

    return output

def download_model(model_name="elyza/Llama-3-ELYZA-JP-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return tokenizer, model

def custom_convert(text):
    # まず全角化して、カタカナと記号を全角化
    text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
    # 次にアルファベットだけを半角に
    text = jaconv.z2h(text, ascii=True, digit=False, kana=False)
    return text


def normalize_columns(columns):
    return (
        columns.str.strip()  # 前後の空白を削除
        .str.replace(r"\s+", "", regex=True)  # 空白をすべて削除
        .str.replace(r"　", "")  # 全角スペースを削除
    )


def generate(target_columns, text, tokenizer, model):

    columns = [
        target_columns, "性別", "身長", "体重", "年齢", "HbA1c", "CRP", "血圧", "体温", "脈拍", "抗血小板薬", "抗凝固薬",
        "スタチン", "糖尿病治療薬", "糖尿病", "喫煙", "飲酒", "診断名", "降圧薬", "プロブレムリスト", "外科治療歴の有無",
        "発症前mRS", "入院前生活場所", "来院時PT-INR"
    ]
    columns = [custom_convert(col) for col in columns]

    replace_rules = {
    r".*喫煙.*": "喫煙",
    r".*飲酒.*": "飲酒",
    r".*抗凝固薬.*": "抗凝固薬",
    r".*抗血小板薬.*": "抗血小板薬",
    r".*スタチン.*": "スタチン",
    r".*糖尿病治療薬.*": "糖尿病治療薬",
    r".*降圧薬.*": "降圧薬",
    r"\(.*?\)": "",
    "（ワルファリン症例）":"",
    "(ワルファリン症例)":"",
    " ":""
}

    response = output_response(DEFAULT_SYSTEM_PROMPT, text, tokenizer, model)
    pattern = r'"(.*?)":\s*"(.*?)"'
    matches = re.findall(pattern, response)
    add_matches = [(target_columns, text)]
    add_matches.extend(matches)
    df_json = pd.DataFrame([dict(add_matches)])

    df_columns = pd.Series(df_json.columns).replace(replace_rules, regex=True)
    df_json.columns = df_columns
    df_json = df_json.loc[:, ~df_json.columns.duplicated(keep='first')]

    # 全角→半角変換を列名のみに適用
    df_json.columns = [custom_convert(col) for col in df_json.columns]

    logging.basicConfig(
        filename='error_log.txt',  # ログファイル名
        level=logging.INFO,        # ログレベル
        format='%(asctime)s - %(message)s'  # ログのフォーマット
    )

    df_json.columns = normalize_columns(pd.Series(df_json.columns))
    columns_to_check = normalize_columns(pd.Series(columns)).tolist()

    existing_columns = set(df_json.columns)
    # 不足している列を確認
    missing_columns = [col for col in columns_to_check if col not in existing_columns]


    # 不足しているカラムを一括追加（`ERR` の追加処理をループ外で行う）
    for col in missing_columns:
        df_json[col] = "ERR"
        # ログに記録
        logging.info(f"ERR Column: {col}, df_json.columns: {df_json.columns.tolist()}")

    return df_json[columns]


# if __name__ == "__main__":
#     main()
