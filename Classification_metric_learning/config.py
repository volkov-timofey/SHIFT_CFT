import torch
import torch.nn as nn
from pytorch_metric_learning import losses
from torchvision import models, transforms

main_dir = '/content/metric_learning_classifire' # корневая папка
zip_file = '/Fruits_Vegetables.zip'
data_dir = '/data'
train_dir = '/train'
test_dir = '/test'
save_path_ = '/checkpoint/'

# wandb
path_wandb = 'volkov-timm/image_recognition/dataset:v0'
filename = 'Fruits_Vegetables.zip'
unpack_path = main_dir + '/data'

rescale_size = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# aug:
train_transform = transforms.Compose([
                transforms.Resize((rescale_size, rescale_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
val_transform = transforms.Compose([
        transforms.Resize((rescale_size, rescale_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

# model:
embeddings = 128
n_classes = 36

# train:
pretrained = True
model = models.resnet18(pretrained=pretrained).to(device)
lr = 3e-4
gamma = 0.5
weight = None
size_average = None
loss_fn = losses.CosFaceLoss(num_classes=n_classes, embedding_size=embeddings).to(device)
classification_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
epochs = 1
batch_size = 64
k_max = 8

# other:
seed = 12345

