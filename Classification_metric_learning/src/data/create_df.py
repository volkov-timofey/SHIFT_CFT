# create df

import glob
from pathlib import Path

import cv2
import pandas as pd
from sklearn import preprocessing


def pivot_df_data(
        data_path: str
) -> pd.DataFrame:
    """
    Create data frame train and test data
    path|str(label)|train or test|int(label)
    """

    p = Path(data_path)

    labels = [x.name for x in p.iterdir()]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)

    # Датафрейм для работы с данными
    for img_ in glob.glob((data_path + '/*/*/*.p*')):
        png_img = cv2.imread(img_)

        # converting png_to_jpg
        cv2.imwrite(
            img_[:-4] + '.jpg',
            png_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )

    df = pd.DataFrame({'image': glob.glob((data_path + '/*/*/*.j*'))})
    df['label'] = df['image'].apply(lambda p: p.split('/')[-2])
    df['split'] = df['image'].apply(lambda p: p.split('/')[-3])
    df['int_label'] = le.fit_transform(df['label'])

    return df
