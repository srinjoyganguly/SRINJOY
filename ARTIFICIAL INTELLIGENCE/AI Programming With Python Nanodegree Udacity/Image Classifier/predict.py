import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
import json
import argparse

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    image_aspect_ratio = image.size[1] / image.size[0]
    image = image.resize((256, int(image_aspect_ratio*256)))
    new_width = image.size[0] / 2
    new_height = image.size[1] / 2
    image = image.crop((new_width - 112, new_height - 112, new_width + 112, new_height + 112))
    image = np.array(image)
    image = image/255
    mean_for_norm = np.array([0.485, 0.456, 0.406])
    std_dev_for_norm = np.array([0.229, 0.224, 0.225])
    image = (image - mean_for_norm) / std_dev_for_norm
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)

def predict(image_path, cnn_model, cuda, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    if cuda:
        cnn_model.cuda()
    else:
        cnn_model.cpu()
    
    image = None
    cnn_model.eval()
    img = Image.open(image_path)
    image = process_image(img)
    if cuda:
        image = image.cuda()
    
    image = Variable(image.unsqueeze(0), volatile=True)
    output = cnn_model.forward(image.float())
    expo = torch.exp(output)
    prob, idx = expo.topk(topk)
    return [j for j in prob.data[0].cpu().numpy()], [cnn_model.idx_to_class[i] for i in idx.data[0].cpu().numpy()]

def chkpt_load(path, cuda):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    
    cnn_model = checkpoint['cnn_model']
    cnn_model.fc = checkpoint['fc']
    cnn_model.load_state_dict(checkpoint['state_dict'])
    cnn_model.class_to_idx = checkpoint['class_to_index']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in cnn_model.parameters():
        param.requires_grad = False
        
    return cnn_model

def retrieve_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_image", action="store", dest="path")
    parser.add_argument("--checkpoint", action="store", dest="checkpoint", default='checkpoint.pth')
    parser.add_argument("--top_k", action="store", dest="top_k", default=5, help="Number of top results you want to view.")
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json", 
                        help="Number of top results you want to view.")
    parser.add_argument("--gpu", action="store", dest="cuda", default=False, help="CUDA True for using the GPU")
    return parser.parse_args()

def basic_function():
    args = retrieve_arguments()
    cuda = args.cuda
    cnn_model = chkpt_load(args.checkpoint, cuda)
    train_dir = 'flowers' + '/train'
    data_transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])])
    image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_training)
    cnn_model.idx_to_class = dict([[n, m] for m, n in image_datasets_train.class_to_idx.items()])
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
    prob, classes = predict(args.path, cnn_model, cuda, topk=int(args.top_k))
    print([cat_to_name[l] for l in classes])
    print(prob)
    print(classes)

basic_function() 