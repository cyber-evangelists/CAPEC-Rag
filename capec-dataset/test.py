
import pandas as pd
from pathlib import Path




def read_file( file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, 
            sep=',',
            encoding='utf-8',
            skipinitialspace=True, index_col=None)
    
    df.columns = df.columns.map(lambda x: x.strip("'\"")) 
    df_reset = df.reset_index(drop=False)

    col_names = df.columns

    df.columns = col_names

    df = df_reset.iloc[:, :-1]

    df.columns = col_names
    
    return df


df = read_file("333.csv")

print(df.columns)