from typing import Tuple

import torch
import torch.nn as nn
import tqdm
from pytorch_metric_learning import losses
from tqdm import tqdm

import config


def train_one_epoch(
        models: dict,
        dataloader: torch.utils.data.DataLoader
) -> Tuple[float, float]:
    """
    Обучение  1 эпохи

    """

    trunk = models["trunk"].train()
    embedder = models["embedder"].train()
    classifier = models["classifier"].train()

    trunk_optimizer = config.optimizer
    embedder_optimizer = config.optimizer
    classifier_optimizer = config.optimizer

    loss_f = config.loss_fn

    # Set the classification loss:
    classification_loss = config.classification_loss

    train_running_loss = 0.0
    train_running_correct = 0
    total_epoch_steps = int(len(dataloader.dataset) / dataloader.batch_size)

    for _, batch in tqdm(enumerate(dataloader), total=total_epoch_steps):
        data, labels = batch
        data, labels = data.to(config.device), labels.to(config.device)

        output_trunk = trunk(data)
        output_embedding = embedder(output_trunk)
        output_classifier = classifier(output_embedding)

        loss = loss_f(output_embedding, labels) \
               + 0.5 * classification_loss(output_classifier, labels)
        train_running_loss += loss.item()

        _, preds = torch.max(output_classifier.data, 1)
        train_running_correct += (preds == labels).sum().item()

        trunk_optimizer.zero_grad()
        embedder_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        loss.backward()

        trunk_optimizer.step()
        embedder_optimizer.step()
        classifier_optimizer.step()

    train_loss = train_running_loss / len(dataloader.dataset)
    train_accuracy = train_running_correct / len(dataloader.dataset)
    return train_loss, train_accuracy
