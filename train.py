import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets,models
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Takes in arguments from the command line
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the folder of images')
    parser.add_argument('--save_dir', type=str, default='checkpoints.pth', help='Path to save the model')
    parser.add_argument('--arch', type=str, default='vgg13', help='Choose architecture: vgg13 or densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Set learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Set number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Set number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

# Load data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=64),
        'test': DataLoader(image_datasets['test'], batch_size=64)
    }
    return image_datasets, dataloaders

# Load model
def load_model(arch, hidden_units):
    try:
        model_fn = getattr(models, arch)
    except AttributeError:
        print(f'Error: {arch} is not a valid model name in torchvision.models.')
        exit()

    model = model_fn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    
    if arch.startswith('vgg'):
        input_units = 25088  
    elif arch.startswith('densenet'):
        input_units = 1024  
    else:
        print(f'Error: Model architecture {arch} is not supported by this script.')
        exit()

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    return model, criterion, optimizer

# Train model with visualization
def train(model, criterion, optimizer, dataloaders, epochs, device):
    print('Model Training:')
    model.to(device)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [],[]

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(dataloaders['train'])
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        valid_loss = 0
        correct_valid = 0
        total_valid = 0
        
        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
        
        valid_loss = valid_loss / len(dataloaders['valid'])
        valid_accuracy = 100 * correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(f'Epoch {epoch+1}/{epochs}.. '
              f'Train loss: {train_loss:.3f}.. '
              f'Validation loss: {valid_loss:.3f}.. '
              f'Train accuracy: {train_accuracy:.3f}%.. '
              f'Validation accuracy: {valid_accuracy:.3f}%')
    
    # Plotting the losses and accuracies
    epochs_range = range(epochs)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, valid_losses, label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, valid_accuracies, label='Validation Accuracy')
    plt.legend(loc='best')
    plt.title('Training and Validation Accuracy')
    
    plt.show()

# Validate model
def validate(model, dataloaders, criterion, device):
    print('Model Validation:')
    model.eval()
    model.to(device)

    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloaders['valid']:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation loss: {valid_loss/len(dataloaders["valid"]):.3f}.. '
          f'Validation accuracy: {accuracy:.3f}%')

# Save model
def save(model, image_datasets, save_dir, arch):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier
    }
    torch.save(checkpoint, save_dir)
    print(f'Model saved to {save_dir}')

# Main function
def main():
    args = get_input_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f'Using {torch.cuda.get_device_name(0)} for training')
    image_datasets, dataloaders = load_data(args.data_dir)
    model, criterion, optimizer = load_model(args.arch, args.hidden_units)
    train(model, criterion, optimizer, dataloaders, args.epochs, device)
    validate(model, dataloaders, criterion, device)
    save(model, image_datasets, args.save_dir, args.arch)

if __name__ == "__main__":
    main()