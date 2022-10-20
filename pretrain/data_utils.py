import pandas as pd
from pathlib import Path
import cv2


def csv2pkl_train(train_path, img_sizes=True, pkl_name="train_df") -> None:
    print("Reading all paths")
    df = pd.read_csv(f'{train_path}/train.csv')
    paths = Path(f'{train_path}').glob('**/*.jpg')
    print("Creating new DataFrame")
    df_path = pd.DataFrame(paths, columns=['path'])
    df_path['path'] = df_path['path'].apply(lambda x: str(x.absolute()))
    df_path['id'] = df_path['path'].apply(
        lambda x: x.split('/')[-1].replace('.jpg', ''))
    print("Merge with the original CSV")
    df = df.merge(df_path, on='id')
    if img_sizes:
        print("Getting additional information for each image")
        df = generate_size_info_df(df)
    print("Savig all informations into a .pkl file")
    df.to_pickle(f'./{pkl_name}.pkl')


def generate_size_info_df(df) -> pd.DataFrame:
    for index, row in df.iterrows():
        img = cv2.imread(str(row['path']))
        h, w, c = img.shape
        df.loc[index, 'height'] = h
        df.loc[index, 'width'] = w
    print(df.columns)
    return df.reset_index().sort_values(by='id')
