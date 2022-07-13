import os
import argparse
import random
import pickle
import joblib
import h5py
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

########################################
######## For Loading Datasets ##########
########################################

def load_dataset_cls(dataset, trim_dataset, num_parties):
    """
    Load the datasets for classification tasks.
    
    Args:
        dataset (str): dataset name
        trim_datastes (int): the number of data points to keep in total
        num_parties (int): the number of parties in the collaboration
    
    Return:
        the loaded dataset and some relevant vars
    """
    
    if dataset == 'MNIST':
        data = load_MNIST()
        # Get a subset of training data for experiment
        num_to_keep = trim_dataset
        # Size (trim_dataset, 1, 28, 28)
        train_images, train_labels = data['train_images'][:num_to_keep], data['train_labels'][:num_to_keep]
        test_images_tensor = torch.tensor(data['test_images'])
        test_labels = data['test_labels']
        dims = (1, 28, 28)
    elif dataset == 'MNIST_baseline' or dataset == 'MNIST_baseline_resized':
        # Resized only for VGG which takes 32 by 32 images
        assert num_parties == 10
        dataset_sizes = [1000, 1000, 1000, 1000, 1000, 1050, 1100, 1150, 1200, 1250]
        num_to_keep = np.sum(dataset_sizes)
        # Size (trim_dataset, 1, 28, 28)
        data = load_MNIST_baseline(dataset_sizes, resize=True if dataset == 'MNIST_baseline_resized' else False)
        train_images, train_labels = data['train_images'], data['train_labels']
        test_images_tensor = torch.tensor(data['test_images'])
        test_labels = data['test_labels']
        dims = (1, 32, 32) if dataset == 'MNIST_baseline_resized' else (1, 28, 28)
    elif dataset == 'MNIST_MNISTM':
        # To only use with val_domain_shift
        mnist_data = load_MNIST_rgb()
        mnistm_data = load_MNISTM()
        each_party = trim_dataset//num_parties
        # Size (num_parties, each_party, 3, 28, 28)
        train_images, train_labels = split_dataset_for_MNIST_MNISTM(args.num_parties, each_party, mnist_data['train_images'], mnist_data['train_labels'], mnistm_data['train_images'])
        test_images = mnistm_data['test_images']
        test_images_tensor = torch.tensor(test_images)/255.
        test_labels = mnist_data['test_labels']
        dims = (3, 28, 28)
    elif dataset == 'CIFAR_10':
        data = load_CIFAR_10()
        # Get a subset of training data for experiment
        num_to_keep = trim_dataset
        # Size (trim_dataset, 3, 32, 32)
        train_images, train_labels = data['train_images'][:num_to_keep], data['train_labels'][:num_to_keep]
        test_images_tensor = torch.tensor(data['test_images'])
        test_labels = data['test_labels']
        dims = (3, 32, 32)
    elif dataset == 'CIFAR_10_baseline':
        assert num_parties == 10
        dataset_sizes = [1000, 1000, 1000, 1000, 1000, 1050, 1100, 1150, 1200, 1250]
        num_to_keep = np.sum(dataset_sizes)
        # Size (trim_dataset, 1, 28, 28)
        data = load_CIFAR_10_baseline(dataset_sizes)
        train_images, train_labels = data['train_images'], data['train_labels']
        test_images_tensor = torch.tensor(data['test_images'])
        test_labels = data['test_labels']
        dims = (3, 32, 32)
    else:
        raise NotImplementedError()
    
    return train_images, train_labels, test_images_tensor, test_labels, dims, num_to_keep


def load_dataset_reg(dataset, split_method):
    """
    Load the datasets for regression tasks.
    
    Args:
        dataset (str): dataset name
        trim_datastes (int): the number of data points to keep in total
        num_parties (int): the number of parties in the collaboration
    
    Return:
        the loaded dataset and some relevant vars
    """
    if dataset == 'ising_baseline' or dataset == 'ising_quantity_aware':
        dataset_sizes = [12, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    elif dataset == 'ising_noise_stability':
        dataset_sizes = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
    else:
        raise NotImplementedError()
    
    dataset_sizes_cumsum = np.cumsum(dataset_sizes)
    num_to_keep = np.sum(dataset_sizes)
    # Size (num_to_keep, 8, 8)
    data = load_isling_baseline(dataset_sizes)
    train_inputs, train_labels = data['train_inputs'], data['train_labels']
    test_inputs, test_labels = data['test_inputs'], data['test_labels']
    test_inputs_tensor = torch.tensor(test_inputs)
    mode = 'reg'
    dims = (8, 8)
    if split_method == 'random':
        permutation_indices = np.random.RandomState(seed=0).permutation(train_inputs.shape[0])
        train_inputs = train_inputs[permutation_indices]
        train_labels = train_labels[permutation_indices]
    else:
        raise NotImplementedError()
        
    if dataset == 'ising_noise_stability':
        noises = []
        for i in range(len(dataset_sizes)):
            noises.append(np.random.RandomState(seed=0).normal(0, 0.05 * i, size=[dataset_sizes[i], *train_inputs.shape[1:]]))
        noises = np.concatenate(noises)
        train_inputs = np.clip(train_inputs + noises, -1, 1)
        
    return train_inputs, train_labels, test_inputs_tensor, test_labels, dims, num_to_keep, dataset_sizes_cumsum


def load_isling(N_train, N_test):
    """
    Load the ising physical model dataset.
    
    Args:
        N_train (int): number of training data samples to load
        N_test (int): number of test data samples to load
    """
    from sklearn.preprocessing import MinMaxScaler
    N = N_train + N_test
    assert N <= 25000 # The file only contains 25000 data samples
    with h5py.File("./data/ising_data.h5",'r') as F:
        inputs = F['data'][:N, ...,0]*1.0
        labels = F['energy'][:N, ...]*1.0
    
    scaler = MinMaxScaler()
    scaler.fit(labels)
    labels = scaler.transform(labels)
        
    train_inputs = inputs[:N_train]
    train_labels = labels[:N_train]
    test_inputs = inputs[-N_test:]
    test_labels = labels[-N_test:]
    
    return {
        'train_inputs': train_inputs,
        'train_labels': train_labels,
        'test_inputs': test_inputs,
        'test_labels': test_labels,
    }


def load_isling_baseline(dataset_sizes):
    """
    Load the ising physical model dataset.
    
    Args:
        dataset_sizes (lsit): list of integers indicating the size of datasets
    """
    N_train = np.sum(dataset_sizes)
    N_test = 10000
    return load_isling(N_train, N_test)

    
def load_MNIST(resize=False):
    """
    Load the MNIST dataset.
    
    Args:
        resize (bool): If True, resize the MNIST images to 32x32
    """
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import torchvision.transforms as transforms

    train_data = datasets.MNIST(
        root = './data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = './data', 
        train = False, 
        transform = ToTensor(),
    )

    if resize:
        # (N, 32, 32)
        resize_transform = transforms.Resize(32)
        train_images = np.expand_dims(resize_transform(train_data.data).numpy()/255, axis=1)
        test_images = np.expand_dims(resize_transform(test_data.data).numpy()/255, axis=1)
    else:
        # (N, 28, 28)
        train_images = np.expand_dims(train_data.data.numpy()/255, axis=1)
        test_images = np.expand_dims(test_data.data.numpy()/255, axis=1)
    train_labels = train_data.targets.numpy()
    test_labels = test_data.targets.numpy()
    
    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'train_data': train_data,
        'test_data': test_data
    }
    
def load_MNIST_baseline(dataset_sizes, resize=False):
    """
    Load the MNIST baseline dataset specified in the paper.
    
    Args:
        dataset_sizes (list): list of integers containing the sizes of each dataset
        resize (bool): if True, resize the MNIST images to 32 by 32
    """
    assert len(dataset_sizes) == 10
    data = load_MNIST(resize=resize)
    train_images, train_labels = data['train_images'], data['train_labels']
    test_images, test_labels = data['test_images'], data['test_labels']

    indices = []
    for party_i in range(10):
        indices.append((train_labels == party_i).nonzero()[0][:dataset_sizes[party_i]])
    indices = np.concatenate(indices, axis=0)

    return {
        'train_images': train_images[indices],
        'train_labels': train_labels[indices],
        'test_images': test_images,
        'test_labels': test_labels,
    }

    
def load_MNIST_rgb():
    """
    Load the RGB version of MNIST dataset.
    """
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    train_data = datasets.MNIST(
        root = './data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = './data', 
        train = False, 
        transform = ToTensor()
    )

    # Process MNIST
    train_images = train_data.data.numpy()
    train_images = np.stack([train_images, train_images, train_images], axis=1)
    train_labels = train_data.targets.numpy()
    test_images = test_data.data.numpy()
    test_images = np.stack([test_images, test_images, test_images], axis=1)
    test_labels = test_data.targets.numpy()
    
    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'train_data': train_data,
        'test_data': test_data,
    }
    

def load_CIFAR_10():
    """
    Load the CIFAR-10 dataset.
    """
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )

    test_data = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    train_images, train_labels = next(iter(train_loader))
    train_images, train_labels = train_images.numpy(), train_labels.numpy()
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    test_images, test_labels = next(iter(test_loader))
    test_images, test_labels = test_images.numpy(), test_labels.numpy()
    
    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels,
    }

    
def load_CIFAR_10_baseline(dataset_sizes):
    """
    Load the baseline dataset for CIFAR_10 defined in the paper.
    
    Args:
        dataset_sizes (list): list of integers indicating the sizes of each dataset
    """
    assert len(dataset_sizes) == 10
    data = load_CIFAR_10()
    train_images, train_labels = data['train_images'], data['train_labels']
    test_images, test_labels = data['test_images'], data['test_labels']

    indices = []
    for party_i in range(10):
        indices.append((train_labels == party_i).nonzero()[0][:dataset_sizes[party_i]])
    indices = np.concatenate(indices, axis=0)

    return {
        'train_images': train_images[indices],
        'train_labels': train_labels[indices],
        'test_images': test_images,
        'test_labels': test_labels,
    }


def load_MNISTM():
    """
    Load the train and test images for the MNISTM dataset.
    Note that test_labels are the same as MNIST, so it is not loaded here.
    MNIST_M dataset created as in https://github.com/pumpikano/tf-dann.
    """
    with open('data/keras_mnistm.pkl', 'rb') as f:
        mnistm = pickle.load(f, encoding='latin1')
    train_images = mnistm['train']
    test_images = mnistm['test']
    
    # Transform a bit to make it diff from original MNIST, so that domain adaptation makes sense
    train_images= np.concatenate([train_images[:,3:,:,:], train_images[:,:3,:,:]], axis=1)
    train_images= np.concatenate([train_images[:,:,3:,:], train_images[:,:,:3,:]], axis=2)
    test_images= np.concatenate([test_images[:,3:,:,:], test_images[:,:3,:,:]], axis=1)
    test_images= np.concatenate([test_images[:,:,3:,:], test_images[:,:,:3,:]], axis=2)
    
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))
    
    return {
        'train_images': train_images,
        'test_images': test_images,
    }
    
    
def split_dataset_for_MNIST_MNISTM(num_parties, each_party, mnist_images, mnist_labels, mnistm_images):
    """
    Create datasets, party 0 to party 9 contains 10% to 100% of mnistm image.
    
    Args:
        num_parties (int): number of parties in the collaboration (we only use 10 in the experiment)
        each_party (int): number of data samples in each party's dataset
        mnist_images (np.ndarray): the mnist images
        mnist_labels (np.ndarray): the correspnding labels of mnist_images
        mnistm_images (np.ndarray): the mnistm images
        
    Return:
        Splitted datasets indexed by each party
    """
    
    # Create datasets, party 0 to party 9 contains 10% to 100% of mnistm image
    mnistm_indices = np.concatenate([[0,], np.cumsum(np.linspace(0.1, 1, num_parties)*each_party).astype(int)])
    mnist_indices = np.concatenate([[0,], np.cumsum(np.linspace(-0.9, 0, num_parties)*each_party).astype(int)])-1

    train_images = []
    train_labels = []

    for i in range(num_parties):
        train_images.append(np.concatenate([mnistm_images[mnistm_indices[i]:mnistm_indices[i+1]],
                                            mnist_images[mnist_indices[i+1]:mnist_indices[i]]]))
        train_labels.append(np.concatenate([mnist_labels[mnistm_indices[i]:mnistm_indices[i+1]],
                                            mnist_labels[mnist_indices[i+1]:mnist_indices[i]]]))
    train_images = np.array(train_images)/255.
    train_labels = np.array(train_labels)
    
    return train_images, train_labels


########################################
########## For Model Training ##########
########################################


def train(model, loaders, loss_func, optimizer, device, num_epochs=100, threshold=10e-5, minimum_epochs=30):
    """
    Train a given model to convergence.
    
    Args:
        model (torch nn): a nn model implemented using torch
        loader (dict): a dictionary of two torch.utils.data.DataLoader, with keys 'train' and 'test'
        loss_func (torch.nn loss): nn loss function
        optimizer (torch.optim optimizer): torch optimizer 
        num_epochs (int): max epochs to train
        threshold (float): loss change between epochs < threshold, considered as converged
        minimum_epochs (int): min epochs to train before testing for convergence
    """    
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
    prev_loss = 0.0
    count = 0
    
    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        epoch_correct = 0
        
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            b_y = Variable(labels).to(device)   # batch y

            output = model(b_x)
            loss = loss_func(output, b_y)
            epoch_loss += loss.item()

            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
                
        # Checking convergence
        epoch_loss /= total_step
        if 0 < torch.abs(torch.tensor(prev_loss) - epoch_loss) < threshold and epoch > minimum_epochs:
            print('Used {} epochs; Loss difference {}'.format(epoch, torch.abs(torch.tensor(prev_loss) - epoch_loss)))
            return
        else:
            prev_loss = epoch_loss
            
            
def test(model, loaders, loss_func, device):
    """
    Test a trained classification model on the test set.
    
    Args:
        model (torch nn): a trained nn model implemented using torch
        loader (dict): a dictionary of two torch.utils.data.DataLoader, with keys 'train' and 'test'
        loss_func (torch.nn loss): nn loss function
    
    Return:
        accuracy: test accuracy
        loss: test loss
    """
    model.eval()
    with torch.no_grad():
        losses = []
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output = model(images.to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            losses.append(loss_func(test_output, labels.to(device)).item())
            correct += ((pred_y == labels.to(device)).sum()).item()
            total += labels.size(0)
        accuracy = correct/total
        loss = np.mean(losses)
    print('Test Accuracy of the model on the 10000 test images: {}, Loss: {}'.format(accuracy,loss))
    return accuracy, loss


def test_reg(model, loaders, loss_func, device):
    """
    Test a trained regression model on the test set.
    
    Args:
        model (torch nn): a trained nn model implemented using torch
        loader (dict): a dictionary of two torch.utils.data.DataLoader, with keys 'train' and 'test'
        loss_func (torch.nn loss): nn loss function
    
    Return:
        accuracy: test accuracy
        loss: test loss
    """
    model.eval()
    with torch.no_grad():
        losses = []
        total = 0
        for inputs, labels in loaders['test']:
            test_output = model(inputs.to(device))
            losses.append(loss_func(test_output, labels.to(device)).item())
        loss = np.mean(losses)
    print('Test loss of the model on the test data: {}'.format(loss))
    return loss
