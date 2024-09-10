import pandas as pd
import os

def save_as_csv(csv_file_path, dict_data):
    # 如果 CSV 文件不存在，则创建一个新文件并写入新表头
    try:
        df = pd.read_csv(csv_file_path)
        names = df.columns.tolist()
        extra_keys = list(set(dict_data.keys()) - set(names))
        lack_keys = list(set(names) - set(dict_data.keys()))
        df = pd.read_csv(csv_file_path,header=None,names=names+extra_keys)
        df.to_csv(csv_file_path,index=False)
        
        for lack_key in lack_keys:
            dict_data[lack_key] = None
        
        df.loc[len(df)] = dict_data.values()
        df = df.drop(index=0)
        
        # sort the columns
        values = [dict_data[key] for key in df.columns.tolist()]
        df.loc[len(df)] = values
        df.to_csv(csv_file_path, index=False)
        
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=dict_data)
        df.loc[len(df)] = dict_data.values()
        df.to_csv(csv_file_path, index=False)
        
    except FileNotFoundError:
        df = pd.DataFrame(columns=dict_data)
        df.loc[len(df)] = dict_data.values()
        df.to_csv(csv_file_path, index=False)

def debug():
    
    # 示例用法
    csv_file_path = "example.csv"
    # new_header = ["NewName", "NewAge", "NewCountry","NewCity"]
    data = {

        "NewAge": 30,
        "NewName": "John",
        # "NewCountry": "China",        
        # "NewCity": "Beijing",
        "test": "test",

    }
    save_as_csv(csv_file_path, data)
    
if __name__ == "__main__":
    debug()