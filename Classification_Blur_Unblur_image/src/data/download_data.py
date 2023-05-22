import shutil


def data_extract(filename, extract_dir):
    """
    Скачивание датасета
    Распаковка архива в необходимую директорию
    """

    import wandb
    run = wandb.init()
    artifact = run.use_artifact('volkov-timm/requrements_hyperparameters/my-dataset:v0', type='dataset')
    artifact.download('artifacts')

    shutil.unpack_archive(filename, extract_dir)
