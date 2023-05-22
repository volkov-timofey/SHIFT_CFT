import torch
import torch.nn as nn
from torchvision import models, transforms

main_dir = './'
zip_file = './data/dataset/shift-cv-winter-2023.zip'
data_dir = './data/dataset'
train_dir = '/train/train'
test_dir = '/test/test'
train_file = '/train.csv'
save_path_ = './checkpoint/'

rescale_size = 224

data_modes = ['train', 'val', 'test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

# aug:
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# train:
pretrained = True
model = models.resnet18(pretrained=pretrained).to(device)
lr = 3e-4
gamma = 0.5
weight = None
size_average = None
criterion = nn.CrossEntropyLoss(weight=weight, size_average=size_average)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
epochs = 1
batch_size = 32

# model:
num_features = 512
n_classes = 2

# other:
seed = 12345

# split
test_size = 0.25
