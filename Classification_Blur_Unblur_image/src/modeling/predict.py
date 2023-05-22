import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import config


def predict_proba(model: nn.Sequential, data_loader: DataLoader) -> np.array[float]:
    """
    рассчет вероятности принадлежности к определенному классу
    blur/unblur
    """

    with torch.no_grad():
        logits = []

        for inputs in data_loader:
            inputs = inputs.to(config.device)
            model.eval()
            outputs = model(inputs).to(config.device_cpu)
            logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs

def predict(dataset: Dataset) -> tuple[list[int], list[int]]:
    """
    Для рандомных индексов предсказываем принадлежность к классу
    """

    idxs = list(map(int, np.random.uniform(0, 650, 20)))  # индексы  20 рандомных изображений
    imgs = [dataset[id][0].unsqueeze(0) for id in idxs]  # изображения

    # вероятности предсказаний к определенному классу
    probs_ims = predict_proba(config.model, imgs)

    probs_img = [np.round(prob[1], 1) for prob in probs_ims]

    y_pred = np.argmax(probs_ims, -1)

    preds_class = [i for i in y_pred]

    return preds_class, idxs


def roc_auc_round(dataset: Dataset) -> float:
    """
    рассчет метрики roc_auc, c округлением
    """

    preds_class, idxs = predict(dataset)

    actual_labels = [dataset[id][1] for id in idxs]
    roc_auc = roc_auc_score(actual_labels, preds_class, average='weighted')
    roc_auc = np.round(roc_auc,2)

    return roc_auc
