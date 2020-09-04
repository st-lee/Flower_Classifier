import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import time
import train_args
args = train_args.get_args()
print(args)


############### Label mapping ###################
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


################ Load the data ###################
data_dir = args.data_dir
train_dir = data_dir + '/train'
val_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


val_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
val_data = datasets.ImageFolder(val_dir, transform = val_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64)



################ Building and Training the Classifier ###################
# My initial variables
arch = args.arch
lr = args.learning_rate
hidden_layer = args.hidden_units
gpu = args.gpu #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = args.epochs
dropout = args.dropout
save_dir = args.save_dir


# Built my classifier
def Classifier(arch = 'vgg16', dropout = 0.5, hidden_layer = 1024):
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_layer = 25088
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained = True)
        input_layer = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_layer = 1024
        
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    My_Classifier = nn.Sequential(nn.Linear(input_layer, hidden_layer),
                           nn.ReLU(),
                           nn.Dropout(dropout),
                           nn.Linear(hidden_layer, 512),
                           nn.ReLU(),
                           nn.Dropout(dropout),
                           nn.Linear(512, 102),
                           nn.LogSoftmax(dim = 1)
                           )
    model.classifier = My_Classifier
    return model

model = Classifier()



# Set criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

if gpu == True:
    device = 'cuda'
else:
    device = 'cpu'

print("Now the device is: {}".format(device))
model.to(device);


# Training and Validation
steps = 0
print_every = 32
running_loss = 0
start = time.time()
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            val_loss = 0
            acc = 0
            model.eval()
            with torch.no_grad():
                
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    val_loss += batch_loss.item()
                    
                    # compute accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    acc += torch.mean(equals.type(torch.FloatTensor)).item()
        
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(train_loader):.3f} | "
                  f"Valdation loss: {val_loss/len(val_loader):.3f} | "
                  f"Valdation accuracy: {acc/len(val_loader):.3f} | "
                  f"During: {time.time() - start:.3f} sec")
            
            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))
            running_loss = 0
            model.train()
            start = time.time()



################## Testing my newwork ####################
# TODO: Do validation on the test set
model.eval()

with torch.no_grad():
    model.to(device)
    test_loss = 0
    test_acc = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        test_loss += criterion(logps, labels)
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        test_acc += torch.mean(equals.type(torch.FloatTensor)).item()
        
    print(f"Testset accuracy: {test_acc/len(test_loader):.3f}")


################## Save the checkpoint  ####################
# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch' : arch,
              'lr' : lr,
              'hidden_layer' : hidden_layer,
              'device' : device,
              'epochs' : epochs,
              'dropout' : dropout,
              'state_dict' : model.state_dict(),
              'class_to_idx' : model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')








