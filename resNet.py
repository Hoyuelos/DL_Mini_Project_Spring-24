import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt


# path for no_label_test set pkl file
test_batch_path='data/cifar_test_nolabels.pkl'

# path for CIFAR data set
cifar_dataset_path = './data'



# Defining Resnet Block
class Resnet_block(nn.Module):
  def __init__(self,in_channels,out_channels,stride=1):
    super(Resnet_block,self).__init__()
    self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=stride,padding=1,bias=False)
    self.bn1= nn.BatchNorm2d(out_channels)
    self.relu1=nn.ReLU()
    
    self.conv2=nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
    self.bn2=nn.BatchNorm2d(out_channels)
    self.relu2=nn.ReLU()

    self.residual=nn.Sequential()
    if stride!=1 or in_channels!=out_channels:
      self.residual=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=stride, bias=False),
                                  nn.BatchNorm2d(out_channels)
                                  )

  def forward(self,x):
    
    out=self.conv1(x)
    out=self.bn1(out)
    out= self.relu1(out)

    out =self.conv2(out)
    out =self.bn2(out)

    out +=self.residual(x)
    out= self.relu2(out)
    return out
  

# Defining Custom Resnet Model
class custom_Resnet(nn.Module):
  def __init__(self,block,n_start_filters,layers,num_classes,dropout_prob=0.5):
    super(custom_Resnet,self).__init__()
    self.in_channels=n_start_filters
    self.layer1=nn.Sequential(
    nn.Conv2d(3,n_start_filters,kernel_size=3,bias=False,padding=1),
    nn.BatchNorm2d(n_start_filters),
    nn.ReLU(inplace=True),
    )
    self.layer2=self.make_layer(block,n_start_filters,layers[0],stride=1)
    self.layer3=self.make_layer(block,n_start_filters*2,layers[1],stride=2)
    self.layer4=self.make_layer(block,n_start_filters*4,layers[2],stride=2)
    self.dropout = nn.Dropout(dropout_prob)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(self.in_channels, num_classes)

  
  def make_layer(self,block,out_channels,n_blocks,stride):
    layers=[]
    layers.append(block(self.in_channels,out_channels,stride))
    self.in_channels=out_channels
    layers.extend([block(out_channels,out_channels) for i in range(1,n_blocks)])
    return nn.Sequential(*layers)    

  def forward(self, x):    
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.dropout(out)
    out = self.fc(out)
    return out

# initializing our custom model
model=custom_Resnet(Resnet_block,32,[13,13,13],10,0.5)



# method to load pickle file
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# DataSet for No_Label_test set
class CustomCIFAR10TestDataset(Dataset):
    def __init__(self, test_data, transform=None):
        self.test_images = test_data[b'data']
        self.ids = test_data[b'ids']
        self.transform = transform

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        image = self.test_images[idx].reshape(3, 32, 32).astype(np.float32) / 255.0
        image_id = self.ids[idx]

        if self.transform:
            image = self.transform(torch.tensor(image))

        return image, image_id

# Loading CIFAR_train set and no_label_test set
def load_cifar10(batch_size, augment=True, test_batch_path=None):
    # Transformations for the training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalization for the test images
    transform_test = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 training data
    train_set = torchvision.datasets.CIFAR10(root=cifar_dataset_path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = None
    if test_batch_path:
        # Load custom test batch
        test_data = unpickle(test_batch_path)
        test_dataset = CustomCIFAR10TestDataset(test_data=test_data, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader



transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

# cifar test set for validation
cifar_test_set = torchvision.datasets.CIFAR10(root=cifar_dataset_path, train=False, download=True, transform=transform_test)

cifar_test_loader = DataLoader(cifar_test_set, batch_size=128, shuffle=False, num_workers=2)



# method to train our model, validate on Validation test and creating prediction (csv) for no_label_test set
def train_and_validate(model, criterion, optimizer, train_loader, test_loader, start_epoch, end_epoch, device='cuda'):
    model = model.to(device)  # Move model to the appropriate device

    best_accuracy = 0.0  # Track the best accuracy to save the best model
    optimizer_name = type(optimizer).__name__
    lr = optimizer.param_groups[0]['lr']
    max_lr = 0.1  # maximum learning rate
    div_factor = 10  # factor to divide the maximum learning rate by
    pct_start = 0.3  # percentage of the cycle used for increasing the learning rate
    epochs_to_run = end_epoch - start_epoch
    lr_scheduler = OneCycleLR(optimizer, max_lr=max_lr, div_factor=div_factor, 
                       pct_start=pct_start, cycle_momentum=False,steps_per_epoch=400 ,epochs=epochs_to_run)
    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    for epoch in range(start_epoch, end_epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        test_acc = 0
        test_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss/len(train_loader)
        print(f'Epoch {epoch}, Loss: {running_loss/len(train_loader)}, Accuracy: {epoch_accuracy}%')

        #evaluaing on cifar test set  (acting as validation data for us)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in cifar_test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        print('Accuracy of the network on the cifar test testset: %d %%' % (test_acc))
        test_loss = test_loss/len(cifar_test_loader)

        all_train_losses.append(epoch_loss)
        all_train_accuracies.append(epoch_accuracy)
        all_test_losses.append(test_loss)
        all_test_accuracies.append(test_acc)

        # saving checkpoint after every 50 epochs
        if epoch%50 == 0:            
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': all_train_losses,
                'train_accuracies': all_train_accuracies,
                'test_losses': all_test_losses,
                'test_accuracies': all_test_accuracies
            }
            torch.save(checkpoint, f'checkpoint/checkpoint_customResNet_{optimizer_name}_epochs_{epoch}_lr_{lr:.6f}_acc_{test_acc}.pth')
        

    
    # Saving final checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': all_train_losses,
        'train_accuracies': all_train_accuracies,
        'test_losses': all_test_losses,
        'test_accuracies': all_test_accuracies
    }
    torch.save(checkpoint, f'checkpoint/checkpoint_customResNet_{optimizer_name}_epochs_{epoch}_lr_{lr:.6f}_acc_{test_acc}.pth')


    # Evaluation on the custom test set
    model.eval()
    predictions = []
    ids = []  # Collect IDs
    with torch.no_grad():
        for inputs, ids_batch in test_loader:  # Corrected to handle inputs and ids
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().tolist())  # Convert to list for easier handling
            ids.extend(ids_batch.tolist())  # Collect IDs

    # Convert predictions and IDs to a DataFrame and save as CSV
    predictions_df = pd.DataFrame({
        'ID': ids,  # Use collected IDs
        'Labels': predictions  # Use collected predictions
    })
    predictions_df.to_csv(f'test_nolabels_csv/test_predictions_customResNet_{optimizer_name}_{epoch}_lr_{lr:.6f}_acc_{test_acc}.csv', index=False)
    print("Predictions saved to test_predictions.csv.")


# plotting the loss and accuracy from the saved checkpoint
def plot_saved_metrics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    train_losses = checkpoint['train_losses']
    train_accuracies = checkpoint['train_accuracies']
    test_losses = checkpoint['test_losses']
    test_accuracies = checkpoint['test_accuracies']
    
    epochs = range(len(train_losses))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    n_epochs = 500
        

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=1e-4)

    # Load data
    train_loader, test_loader = load_cifar10(batch_size=batch_size, augment=True, test_batch_path=test_batch_path)
    
    summary(model.to(device), (3, 32, 32))

    start_epoch=1                  
    end_epoch= start_epoch + n_epochs
    print('start epoch {s}, end epoch {e}'.format(s=start_epoch, e=end_epoch-1))

    train_and_validate(model, criterion, optimizer, train_loader, test_loader, start_epoch, end_epoch, device=device)

    plot_saved_metrics('checkpoint/checkpoint_customResNet_SGD_epochs_500_lr_0.100000_acc_96.49.pth')



if __name__ == '__main__':
  main()