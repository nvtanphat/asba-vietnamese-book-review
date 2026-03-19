import pandas as pd
import os
import json

def convert_json_to_csv(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".json") and "clean" in filename:
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            csv_filename = filename.replace(".json", ".csv")
            df.to_csv(os.path.join(folder, csv_filename), index=False, encoding="utf-8-sig")
            print(f"Converted {filename} to {csv_filename}")

if __name__ == "__main__":
    convert_json_to_csv("d:/DataPreprocessing/DoAn2/data/processed")
