import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np

# Takes in arguments from the command line
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

# Load the checkpoint
def load(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    return model


# Process the image
def process_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    return img

# Predict the class of the image

def predict(image_path, model, topk=1, gpu=False):
    model.eval()
    if gpu:
        model.to('cuda')
    else:
        model.to('cpu')
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    if gpu:
        image = image.to('cuda')
    else:
        image = image.to('cpu')
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        probs, indices = ps.topk(topk)
        probs = probs.cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[i] for i in indices]
        return probs, classes

# Display the results
def display(image_path, model, topk, cat_to_name, gpu):
    probs, classes = predict(image_path, model, topk, gpu)
    names = [cat_to_name[i] for i in classes]
    print('Probabilities:', probs)
    print('Classes:', classes)
    print('Names:', names)
    print('Most likely class:', names[0])

# Main function
def main():
    args = get_input_args()
    model = load(args.checkpoint)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    display(args.image_path, model, args.top_k, cat_to_name, args.gpu)

if __name__ == '__main__':
    main()