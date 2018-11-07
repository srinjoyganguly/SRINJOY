from collections import OrderedDict
import argparse
import numpy as np
import os
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable

def training_network(cnn_model, dataloaders_training, dataloaders_validation, criterion, optimizer, epochs=5, cuda=False):
    print_every = 20
    steps = 0
    if cuda and torch.cuda.is_available:
        cnn_model.to('cuda')
    else:
        cnn_model.cpu()
    
    for e in range(epochs):
        
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders_training):
            
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            steps += 1
            if cuda:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = cnn_model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.data[0]
        
            if steps % print_every == 0:
                cnn_model.eval()
                val_accuracy = 0
                val_loss = 0

                for ii, (images, labels) in enumerate(dataloaders_validation):
                    inputs = Variable(images, requires_grad=True)
                    labels = Variable(labels, requires_grad=True)
                    if cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    
                    outputs = cnn_model.forward(inputs)
                    val_loss += criterion(outputs, labels).data[0]
                    expo = torch.exp(outputs).data
                    equal_comp = (labels.data == expo.max(1)[1])
                    val_accuracy += equal_comp.type_as(torch.FloatTensor()).mean()
               
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(val_loss/len(dataloaders_validation)),
                      "Validation Accuracy: {:.3f}".format(val_accuracy/len(dataloaders_validation)))
            
                running_loss = 0
                cnn_model.train()
            
def testing_model(cnn_model, dataloaders_test, cuda=False):
    cnn_model.eval()
    test_accuracy = 0
    if cuda:
        cnn_model.cuda()
    else:
        cnn_model.cpu()
    for ii, (images, labels) in enumerate(dataloaders_test):
        inputs = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if cuda:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        outputs = cnn_model.forward(inputs)
        expo = torch.exp(outputs).data
        equal_comp = (labels.data == expo.max(1)[1])
        test_accuracy += equal_comp.type_as(torch.FloatTensor()).mean()    

    print("Testing Accuracy: {:.3f}".format(test_accuracy/len(dataloaders_test)))
    
def parsing_data(path_of_data):
    train_dir = path_of_data + '/train'
    valid_dir = path_of_data + '/valid'
    test_dir = path_of_data + '/test'
    batch_size = 32
    data_transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])])

    data_transforms_validation = transforms.Compose([transforms.Resize(224),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                                          [0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])


    image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_training)
    image_datasets_val = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)
    image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)


    dataloaders_training = torch.utils.data.DataLoader(image_datasets_train, batch_size=batch_size, shuffle=True)
    dataloaders_validation = torch.utils.data.DataLoader(image_datasets_val, batch_size=batch_size)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=batch_size)
    
    return data_transforms_training, data_transforms_validation, data_transforms_test, image_datasets_train, image_datasets_val,           image_datasets_test, dataloaders_training, dataloaders_validation, dataloaders_test 

def basic_function():
    args = retrieve_arguments()
    input_layers = None
    output_layers = None
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_dir = os.path.join(args.save_dir, 'checkpoint.pth')   
    
    if args.cnn_model == 'densenet121':
        input_layers = 1024
        output_layers = 102
        cnn_model = models.densenet121(pretrained=True)
        for param in cnn_model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, 1000)),
                                               ('relu', nn.ReLU()),
                                               ('dropout2', nn.Dropout(0.3)),
                                               ('fc2',  nn.Linear(1000, 1500)),
                                               ('relu2',  nn.ReLU()),
                                               ('fc3', nn.Linear(1500, 102)),
                                               ('output', nn.LogSoftmax(dim = 1)),
                                               ]))  
        cnn_model.classifier = classifier
        data_transforms_training, data_transforms_validation, data_transforms_test, image_datasets_train, image_datasets_val, image_datasets_test, dataloaders_training, dataloaders_validation, dataloaders_test =  parsing_data(args.path_of_data) 
        if args.cuda:
            cnn_model = cnn_model.cuda()
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(cnn_model.classifier.parameters(), lr=args.lr) 
        training_network(cnn_model, dataloaders_training, dataloaders_validation, criterion = criterion, optimizer = optimizer, epochs=int(args.epochs), cuda=args.cuda)
        testing_model(cnn_model, dataloaders_test, cuda=args.cuda)
        checkpoint = {'optimizer': optimizer.state_dict(),
                      'epochs': args.epochs,
                      'cnn_model': cnn_model,
                      'classifier': classifier,
                      'class_to_index': image_datasets_train.class_to_idx,
                      'batch_size': 32,
                      'learning_rate': args.lr,
                      'input_size': input_layers,
                      'state_dict': cnn_model.state_dict(),
                      'output_size': output_layers,
                      'data_transforms_validation': data_transforms_validation,
                      'data_transforms_test': data_transforms_test}
        torch.save(checkpoint, 'checkpoint.pth')
    elif args.cnn_model == 'resnet18':
        input_layers = 512
        output_layers = 102
        cnn_model = models.resnet18(pretrained=True)
        for param in cnn_model.parameters():
            param.requires_grad = False
        fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(512, 350)),
                                               ('relu', nn.ReLU()),
                                               ('dropout2', nn.Dropout(0.3)),
                                               ('fc2',  nn.Linear(350, 102)),
                                               ('output', nn.LogSoftmax(dim = 1)),
                                               ]))   
        cnn_model.fc = fc
        data_transforms_training, data_transforms_validation, data_transforms_test, image_datasets_train, image_datasets_val, image_datasets_test, dataloaders_training, dataloaders_validation, dataloaders_test =  parsing_data(args.path_of_data)  
        if args.cuda:
            cnn_model = cnn_model.cuda()
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(cnn_model.fc.parameters(), lr=args.lr)
        training_network(cnn_model, dataloaders_training, dataloaders_validation, criterion = criterion, optimizer = optimizer, epochs=int(args.epochs), cuda=args.cuda)
        testing_model(cnn_model, dataloaders_test, cuda=args.cuda)
        checkpoint = {'optimizer': optimizer.state_dict(),
                      'epochs': args.epochs,
                      'cnn_model': cnn_model,
                      'fc': fc,
                      'class_to_index': image_datasets_train.class_to_idx,
                      'batch_size': 32,
                      'learning_rate': args.lr,
                      'input_size': input_layers,
                      'state_dict': cnn_model.state_dict(),
                      'output_size': output_layers,
                      'data_transforms_validation': data_transforms_validation,
                      'data_transforms_test': data_transforms_test}
        torch.save(checkpoint, 'checkpoint.pth')
    
def retrieve_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", action="store", dest="save_dir", default="." , help = " To save checkpoints")
    parser.add_argument("--model", action="store", dest="cnn_model", default="densenet121" , help = "Architechture('densenet121' or     'resnet18')")
    parser.add_argument("--learning_rate", action="store", dest="lr", default=0.001 , help = "Learning rate")
    parser.add_argument("--hidden_units", action="store", dest="hidden_units", default=512 , help = "Number of hidden units")
    parser.add_argument("--epochs", action="store", dest="epochs", default=5 , help = "Number of epochs")
    parser.add_argument("--gpu", action="store", dest="cuda", default=False , help = "Use CUDA True for training")
    parser.add_argument("--path_of_data", action="store", dest="path_of_data", default='flowers')
    
    return parser.parse_args()
    
basic_function()