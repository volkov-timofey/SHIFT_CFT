import torch
from torch.utils.data import Dataset

import config


def eval_epoch(models: dict,
               val_loader: Dataset) -> tuple[float, float, float]:
    """
    Валидация одной эпохи
    """

    trunk = models["trunk"].eval()
    embedder = models["embedder"].eval()
    classifier = models["classifier"].eval()

    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    # необходимы для рассчета mAP@K
    """
    K выбирал минимальное значение семплов в классе
    на выборке
    """
    last_label = None
    flag = True
    k = 1
    count_true = 0
    list_p_k = []
    list_ap_k = []

    for data, labels in val_loader:
        data = data.to(config.device)
        labels = labels.to(config.device)

        if last_label != labels:
            k = 1
            count_true = 0
            flag = True
            list_p_k = []
            last_label = labels
        elif k < config.k_max:
            k += 1
        else:
            # рассчет AP@K для каждого класса
            list_ap_k.append(sum(list_p_k) / len(list_p_k))
            flag = False

        with torch.set_grad_enabled(False):

            output_trunk = trunk(data)
            output_embedding = embedder(output_trunk)
            output_classifier = classifier(output_embedding)

            loss = config.loss_fn(output_embedding, labels) \
                   + 0.5 * config.classification_loss(output_classifier, labels)

            preds = torch.argmax(output_classifier, 1)

            if flag:
                if preds == labels:
                    count_true += int(preds == labels)
                    list_p_k.append(count_true / k)
                else:
                    list_p_k.append(0)

        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(preds == labels)
        processed_size += data.size(0)

    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size

    # mAP@K для всех классов
    map_k = sum(list_ap_k) / len(list_ap_k)  # mAP@K

    return val_loss, val_acc, map_k
