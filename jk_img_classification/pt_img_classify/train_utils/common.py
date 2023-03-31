import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


dataset = '../data/mask_data_set_gray'
model_dir = '../models'
model_name = 'mask'
visualise_dir = '../visualise'
bs = 3  # batch_size
num_epochs = 3

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'validate')
test_directory = os.path.join(dataset, 'test')

num_classes = len(os.listdir(valid_directory))
print('num of classes {}'.format(num_classes))

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}


# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])


# Create iterators for the Data loaded using DataLoader module
train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)

print('sizes train {}  validate {} test {}'.format(train_data_size, valid_data_size, test_data_size))


def get_3d_gray_img(src_file):
    img = cv2.imread(src_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray_3d
