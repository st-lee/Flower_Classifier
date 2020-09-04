import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import pred_args
args = pred_args.get_args()
print(args)


#################### Input args #####################
img_path = args.img_path
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

############## Loading the checkpoint ###############
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(file_path):
    checkpoint = torch.load(file_path)
    arch = checkpoint['arch']
    lr = checkpoint['lr']
    hidden_layer = checkpoint['hidden_layer']
    #device = checkpoint['device']
    epochs = checkpoint['epochs']
    dropout = checkpoint['dropout']
    classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained = True)
    elif arch == 'densenet121':
        model = models.densenet11(pretrained = True)
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
   
    return model

# Reload the modle which we trained before
reload_model = load_model(checkpoint)



############## Inference and Classification ###############
# Image Precessing
def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(img_path)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    #im.crop((left, top, right, bottom))
    
    img_nparr = np.array(img)
    img_nparr = img_nparr / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_nparr = (img_nparr - mean) / std
    img_nparr = img_nparr.transpose((2, 0, 1))
    return img_nparr



###################### Class Prediction ######################
def predict(image_path = img_path, model = reload_model, top_k = top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    image = process_image(image_path)
    img = torch.zeros(1, 3, 224, 224) 
    img[0] = torch.from_numpy(image)
    
    if gpu == True: 
        device = 'cuda'
    else:
        device = 'cpu'
   
    model.to(device)
    model.eval()
    img = img.to(device)
    #Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)
        
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k)
    probs = top_p.cpu().numpy()[0]
    idx = top_class.cpu().numpy()[0]
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in idx]
    
    return probs, classes



# TODO: Implement the code to predict the class from an image file
probs, classes = predict(img_path, reload_model, top_k)
print(probs)
print(classes)

# Print the most likely name of the flower:
top_5 = [cat_to_name[str(i)] for i in classes]
print('\n The name of this flower is most likely called: {}\n'. format(top_5[0]))







