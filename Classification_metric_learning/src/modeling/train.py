import wandb
from tqdm import tqdm

from src.modeling.eval_one_epoch import eval_epoch
from src.modeling.fit_one_epoch import train_one_epoch


def train(
        train_loader,
        val_loader,
        models: dict,
        epochs: int
) -> list[tuple[float, float, float, float, float]]:
    """
    Обучение модели, логирование гиперпараметров,
    сохранение наилучшей модели
    """

    wandb.init(project='HW_metric-learning')

    history = []  # сохраняем данные о loss и accuracy для train и val

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} \
    map_k {map_k:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(models,
                                                    train_loader)

            print("loss", train_loss)

            val_loss, val_acc, map_k = eval_epoch(models, val_loader)

            history.append((train_loss, train_acc, val_loss, val_acc, map_k))

            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "epoch": epoch,
                       "train_acc": train_acc,
                       "val_acc": val_acc,
                       "map_k": map_k
                       })

            pbar_outer.update(1)
            tqdm.write(log_template.format(
                ep=epoch + 1,
                t_loss=train_loss,
                v_loss=val_loss,
                t_acc=train_acc,
                v_acc=val_acc,
                map_k=map_k
            ))

    return history
