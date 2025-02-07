import re
import pandas as pd
from datetime import datetime
from run_llm import download_model, generate
from tqdm import tqdm

def main():

    tokenizer, model = download_model(model_name="elyza/Llama-3-ELYZA-JP-8B")
    target_columns = "text"
    df = pd.read_csv("data/test_sample100.csv")
    print(df.head(3))

    for i in tqdm(range(len(df))):
        df_json = generate(target_columns, df[target_columns][i], tokenizer, model)
        if i == 0:
            output_df = df_json
        else:
            output_df = pd.concat([output_df, df_json],axis=0)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    output_df.to_csv(f"output_df_{timestamp}.csv", index=False)


if __name__ == "__main__":
    main()