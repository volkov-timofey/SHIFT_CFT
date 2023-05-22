from src.data import parsing_json
from src.utils import choose_file, save_file


def main():
    files = parsing_json(choose_file())

    for name in files:
        save_file(files[name], name)


if __name__ == "__main__":
    main()
