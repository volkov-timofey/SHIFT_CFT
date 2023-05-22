import tkinter.filedialog as fd
import tkinter.messagebox as mb
from tkinter import *


def choose_file() -> str:
    """
    Функция для выбора файла, из диалогового окна,
    в случае "отмена", запросит путь из консоли
    Result: строка содержащая путь до файла
    """

    root = Tk()
    filetypes = (("Log", "*.log *.txt"),
                 ("Любой", "*"))
    filename = fd.askopenfilename(title="Выберите файл с логами",
                                  filetypes=filetypes)
    root.destroy()

    if filename:
        return filename
    else:
        return str(input('Введите путь до файла с логами: '))


def show_info(filename: str):
    """
    Диалоговое окно с информацией об успешном сохранении
    """

    msg = f"Файл {filename} успешно сохранен."
    mb.showinfo("Информация", msg)


def save_file(df: pd.DataFrame, filename: str):
    """
    Функция сохраняет полученный файл
    """

    root = Tk()

    new_file = fd.asksaveasfile(
        mode='w',
        title=f"Сохранить файл {filename}",
        defaultextension=".csv",
        filetypes=(
            ("Файл данных", "*.csv"),
            ("Любой", "*")
        )
    )
    root.destroy()
    if new_file:
        df.to_csv(new_file)
        show_info(filename)

    else:
        df.to_csv(str(input('Введите только путь для сохранения файлов: ') /
                      + f'{filename}.csv'))
        show_info(filename)
