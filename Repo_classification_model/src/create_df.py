import glob
import pandas as pd
from sklearn import preprocessing
from pathlib import Path
import config

def pivot_df_data(
    train_path: str
) -> pd.DataFrame:
    '''
    Create data frame train and test data
    path|str(label)|train or test|int(label)
    '''
    
    p = Path(config.path_data) #.replace('\\','\/')

    labels = [x.name for x in p.iterdir()]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)

    # Датафрейм для работы с данными
    df = pd.DataFrame({'image': glob.glob((config.path_data + '/*/*/*.png'))})
    df['label'] = df['image'].apply(lambda p: p.split('/')[-2])
    df['split'] = df['image'].apply(lambda p: p.split('/')[-3])
    df['int_label'] = le.fit_transform(df['label'])

    return df