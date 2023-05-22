import json
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm


@dataclass
class AmtCredit:
    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float


@dataclass
class PosCashBalanceIDs:
    SK_ID_PREV: int
    SK_ID_CURR: int
    NAME_CONTRACT_STATUS: str


COLUMNS_POS_CASH_BALANCE = ['SK_ID_PREV',
                            'SK_ID_CURR',
                            'MONTHS_BALANCE',
                            'CNT_INSTALMENT',
                            'CNT_INSTALMENT_FUTURE',
                            'NAME_CONTRACT_STATUS',
                            'SK_DPD',
                            'SK_DPD_DEF']

COLUMNS_BUREAU = ['SK_ID_CURR',
                  'SK_ID_BUREAU',
                  'CREDIT_ACTIVE',
                  'CREDIT_CURRENCY',
                  'DAYS_CREDIT',
                  'CREDIT_DAY_OVERDUE',
                  'DAYS_CREDIT_ENDDATE',
                  'DAYS_ENDDATE_FACT',
                  'AMT_CREDIT_MAX_OVERDUE',
                  'CNT_CREDIT_PROLONG',
                  'AMT_CREDIT_SUM',
                  'AMT_CREDIT_SUM_DEBT',
                  'AMT_CREDIT_SUM_LIMIT',
                  'AMT_CREDIT_SUM_OVERDUE',
                  'CREDIT_TYPE',
                  'DAYS_CREDIT_UPDATE',
                  'AMT_ANNUITY']


def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Заменяет колонку со словарем на несколько колонок.

    Имена новых колонок - ключи словаря
    Значения в новых колонок - значения по заданному ключу в словаре
    """
    return pd.concat(
        [
            df,
            pd.json_normalize(df[column]),
        ],
        axis=1
    ).drop(columns=[column])


def concat_class_in_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает данные из класса
    Добаляет в промежуточную таблицу по CNT_INSTALMENT
    """

    pos_cash_list = []
    column = 'PosCashBalanceIDs'
    for row in df[column]:
        pos_cash_list.append(eval(row).__dict__)

    return pd.concat(
        [
            df,
            pd.DataFrame(pos_cash_list),
        ],
        axis=1
    ).drop(columns=[column])


def parsing_json(log_file: str) -> dict:
    """
    Функция для парсинга файла bureau
    Вход: строка содержащая путь до файла
    Выход: распарсенный DataFrame для дальнейшего сохранения
    """

    with open(log_file, 'r') as file:

        data_bureau = []
        data_pos_cash_balance = pd.DataFrame(columns=COLUMNS_POS_CASH_BALANCE)

        for line in tqdm(file):

            json_dict = json.loads(line)
            name_table = json_dict['type']
            data_table = json_dict['data']

            if name_table == 'bureau':
                name_data_bureau = name_table
                # Добавляем данные из словаря к CREDIT_TYPE
                data_table.update(json_dict['data']['record'])
                # Удаляем лишний элемент словаря
                del data_table['record']
                # Извлекаем class AmtCredit
                # Данные переводим в словарь и соединяем с итоговым словарем
                data_table.update(eval(data_table['AmtCredit']).__dict__)
                del data_table['AmtCredit']
                # Добавляем полученный словарь к списку словарей
                data_bureau.append(data_table)

            if name_table == 'POS_CASH_balance':
                name_data_pos_cash_balance = name_table
                df_pos = normalize_column(pd.DataFrame(data_table),
                                          'records')
                # Промежуточная таблица по столбцу CNT_INSTALMENT
                df_pos = concat_class_in_data(df_pos)
                data_pos_cash_balance = pd.concat([data_pos_cash_balance, df_pos],
                                                  ignore_index=True)

        return {
            name_data_bureau: pd.DataFrame(data_bureau, columns=COLUMNS_BUREAU),
            name_data_pos_cash_balance: data_pos_cash_balance
        }
