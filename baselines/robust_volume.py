"""
    Run python baselines/robust_volume.py from the project root director
    Example usage:
    
    python baselines/robust_volume.py --dataset=MNIST_baseline --model=VGG13 --num_parties=10 --split_method=by_class --seed=0 --gpu=0
    python baselines/robust_volume.py --dataset=CIFAR_10_baseline --model=ResNet18 --num_parties=10 --split_method=by_class --seed=0 --gpu=0
    python baselines/robust_volume.py --dataset=ising_baseline --model=MLP --num_parties=10 --split_method=random --seed=0 --gpu=0
    
    python baselines/robust_volume.py --dataset=ising_quantity_aware --model=MLP --num_parties=10 --split_method=random --seed=0 --gpu=0 --no-loo
    python baselines/robust_volume.py --dataset=ising_noise_stability --model=MLP --num_parties=10 --split_method=random --seed=0 --gpu=0 --no-loo
    python baselines/robust_volume.py --dataset=MNIST_MNISTM --model=CNN --num_parties=10 --split_method=val_domain_shift --trim_dataset=10000 --seed=0 --gpu=0 --no-loo
"""

import sys
sys.path.insert(0, '.')

import time
import os
import argparse
import random
import numpy as np
from scipy import stats
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
from utils import load_dataset_cls, load_dataset_reg, train, test, test_reg

from model.cnn import CNN
from model.resnet import BasicBlock, ResNet
from model.vgg import VGG
from model.mlp import MLP
from model.cnn8 import CNN as CNN8

from math import ceil, floor
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='Data valuation at initialization.')
parser.add_argument('--gpu', help='gpu device index',
                    required=False,
                    type=str,
                    default='4')
parser.add_argument('--seed', help='seed for reproducibility',
                    required=False,
                    type=int,
                    default=0)
parser.add_argument('--dataset', help='dataset to use: MNIST, MNIST_MNISTM',
                    required=True,
                    type=str,
                    default='MNIST')
parser.add_argument('--trim_dataset', help='number of data points to keep for faster training and experiment',
                    required=False,
                    type=int,
                    default=10000)
parser.add_argument('--num_parties', help='number of parties to value',
                    required=True,
                    type=int,
                    default=10)
parser.add_argument('--split_method', help='method to split the dataset: by_class, random',
                    required=False,
                    type=str,
                    default='by_class')
parser.add_argument('--model', help='model to use: CNN, ResNet18',
                    required=True,
                    type=str,
                    default='CNN')
parser.add_argument('-loo', help='whether to compute leave-one-out value',
                    required=False,
                    type=bool,
                    default=True)
parser.add_argument('--no-loo', dest='loo', 
                    action='store_false')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################
########### Dataset methods ###########
#######################################

def set_seed(seed):
    # Reporducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
def get_indices_on_split(train_inputs, train_labels, party_i, split, extra_args={}, loo=True):
    size = train_labels.shape[0]
    if extra_args['given_indices'] != None:
        party_i_indices = extra_args['given_indices'][party_i] if party_i != extra_args['num_parties'] else []
    elif split == 'by_class':
        party_i_indices = (train_labels == party_i).nonzero()[0]
    elif split == 'random':
        party_i_indices = np.random.randint(0, size, size//extra_args['num_parties'])
    else:
        raise NotImplementedError()
    if loo:
        # Set difference for leave-one-out
        if party_i == extra_args['num_parties']:
            party_i_indices = []
        coalition_i_indices = np.setdiff1d(np.arange(0, size), party_i_indices)
    else:
        coalition_i_indices = party_i_indices
    return coalition_i_indices

#######################################
########### Robust Volume #############
#######################################

def scale_normal(dataset):
    """
        Scale  the training to standard normal distribution.
        Args:
            datasets: datasets of length M (tensor)
        
        Returns:
            the standarized training dataset
    """
    dev = dataset.device
    dataset = dataset.cpu()
    N = dataset.shape[0]
    other_dims = dataset.shape[1:]
    scaler = StandardScaler()
    scaler.fit(dataset.reshape(N, -1))
    transformed = scaler.transform(dataset.reshape(N, -1)).reshape(N, *other_dims)
    return torch.tensor(transformed).to(dev)


def extract_features(model, train_inputs, model_name, mode):
    n = train_inputs.shape[0]
    batch_size = 100
    batches = n // batch_size

    features = []
    for i in range(0, n, batch_size):
        if i == batches:
            inputs = train_inputs[i:]
        else:
            inputs = train_inputs[i:i+batch_size]
    
        if ('VGG' in model_name or 'ResNet' in model_name or 'CNN' in model_name) and (mode == 'cls'):
            feature = model(inputs.to(device)).detach()
        elif ('MLP' in model_name or 'CNN8' in model_name) and (mode == 'reg'):
            feature = model.get_activation_before_last_layer(inputs.to(device)).detach()
        else:
            raise NotImplementedError()    
        features.append(feature)
    
    features = torch.cat(features, axis=0)
    return features

def compute_volume(dataset):
    volume = torch.linalg.det(dataset.t() @ dataset)
    return volume

def compute_X_tilde_and_counts(X, omega):
    """
    Compresses the original feature matrix X to X_tilde with the specified omega.

    Returns:
       X_tilde: compressed np.ndarray
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    D = X.shape[1]

    # assert 0 < omega <= 1, "omega must be within range [0,1]."

    m = ceil(1.0 / omega) # number of intervals for each dimension

    cubes = Counter() # a dictionary to store the freqs
    # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
    # value: counts

    Omega = defaultdict(list)
    # Omega = {}
    
    min_ds = torch.min(X, axis=0).values

    # a dictionary to store cubes of not full size
    for x in X:
        cube = []
        for d, xd in enumerate(x - min_ds):
            d_index = floor(xd / omega)
            cube.append(d_index)

        cube_key = tuple(cube)
        cubes[cube_key] += 1

        Omega[cube_key].append(x)

        '''
        if cube_key in Omega:
            
            # Implementing mean() to compute the average of all rows which fall in the cube
            
            Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
            # Omega[cube_key].append(x)
        else:
             Omega[cube_key] = x
        '''
    X_tilde = torch.stack([torch.stack(list(value)).mean(axis=0) for key, value in Omega.items()])

    # X_tilde = stack(list(Omega.values()))

    return X_tilde, cubes


def compute_robust_volume(X_tilde, hypercubes):
    N = len(X_tilde)
    # N = sum([len(X_tilde) for X_tilde in X_tildes])
    alpha = 1.0 / (10 * N) # it means we set beta = 10
    # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))

    volume = compute_volume(X_tilde)
    # robust_volumes = np.zeros_like(volumes)
    # for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
    rho_omega_prod = 1.0
    for cube_index, freq_count in hypercubes.items():
        
        # if freq_count == 1: continue # volume does not monotonically increase with omega
        # commenting this if will result in volume monotonically increasing with omega
        rho_omega = (1 - alpha**(freq_count + 1)) / (1 - alpha)

        rho_omega_prod *= rho_omega

    robust_volumes = (volume * rho_omega_prod)
    return robust_volumes


def main():
    given_indices = None
    cls_datasets = ['MNIST_baseline', 'MNIST_baseline_resized', 'CIFAR_10_baseline', 'MNIST_MNISTM']
    reg_datasets = ['ising_baseline', 'ising_quantity_aware', 'ising_noise_stability']
    
    if args.dataset in cls_datasets:
        train_inputs, train_labels, test_inputs_tensor, test_labels, dims, num_to_keep = load_dataset_cls(args.dataset, args.trim_dataset, args.num_parties)
        args.trim_dataset = num_to_keep
        mode = 'cls'
    elif args.dataset in reg_datasets:
        train_inputs, train_labels, test_inputs_tensor, test_labels, dims, num_to_keep, dataset_sizes_cumsum = load_dataset_reg(args.dataset, args.split_method)
        args.trim_dataset = num_to_keep
        cumsum = np.concatenate([[0], dataset_sizes_cumsum])
        given_indices = [np.arange(cumsum[party_i], cumsum[party_i + 1]) for party_i in range(len(dataset_sizes_cumsum))]
        mode = 'reg'
    else:
        raise NotImplementedError()

    # Reproducibility
    set_seed(args.seed)
    
    # Construct NN model
    in_channels = dims[0]
    
    if 'MNIST' in args.dataset or 'CIFAR' in args.dataset:
        lr = 0.01
    elif 'ising' in args.dataset:
        lr = 0.1
    else:
        raise NotImplementedError()
    retrain_threshold = 1e-6
        
    if args.model == 'CNN':
        if 'CIFAR_10' in args.dataset:
            model = CNN(in_channels=in_channels, linear_dim=dims[2]//2//2).to(device)
        else:
            model = CNN(in_channels=in_channels).to(device)
            retrain_threshold = 1e-6
    elif args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels).to(device)
    elif args.model == 'ResNet21':
        model = ResNet(BasicBlock, [2, 3, 3, 3], in_channels=in_channels).to(device)
    elif args.model == 'ResNet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels).to(device)
    elif args.model == 'VGG11':
        model = VGG('VGG11', in_channels=in_channels).to(device)
    elif args.model == 'VGG13':
        model = VGG('VGG13', in_channels=in_channels).to(device)
    elif args.model == 'VGG16':
        model = VGG('VGG16', in_channels=in_channels).to(device)
    elif args.model == 'MLP':
        model = MLP(in_dim=np.prod(dims), out_dim=1).to(device)
    elif args.model == 'CNN8':
        dims = (1,) + dims
        model = CNN8(in_channels=dims[0], out_dim=1, linear_dim=dims[1]//2//2).to(device)
    else:
        raise NotImplementedError()
    
    if mode == 'cls':
        train_data = TensorDataset(torch.tensor(train_inputs).reshape(-1, *dims).float().to(device), torch.tensor(train_labels).to(device))
        test_data = TensorDataset(test_inputs_tensor.reshape(-1, *dims).float().to(device), torch.tensor(test_labels).to(device))
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        train_data = TensorDataset(torch.tensor(train_inputs).reshape(-1, *dims).float().to(device), torch.tensor(train_labels).float().to(device))
        test_data = TensorDataset(test_inputs_tensor.reshape(-1, *dims).float().to(device), torch.tensor(test_labels).float().to(device))
        loss_fn = torch.nn.MSELoss()
        
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                            batch_size=128, 
                                            shuffle=True),

        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=128, 
                                            shuffle=False),
    }
    optimizer = optim.SGD(model.parameters(), lr = lr if lr!=None else 0.01) 
    
    # Train and save model
    start_time = time.time()
    train(model, loaders, loss_fn, optimizer, device, num_epochs=300, threshold=retrain_threshold, minimum_epochs=300)
    if mode == 'cls':
        accuracy, loss = test(model, loaders, loss_fn, device)
    else:
        loss = test_reg(model, loaders, loss_fn, device)
        accuracy = None
    print('Finished training the full model. Test Loss: {}; Test Acccuracy: {}'.format(loss, accuracy))
    save_path = 'checkpoints/{}_{}_influence_trained_model_seed{}.pt'.format(args.dataset.lower(), args.model.lower(), args.seed)
    torch.save(model, save_path)

    # Extract the embeddings in the learned latent space
    train_inputs_tensor = torch.tensor(train_inputs.reshape(-1, *dims), dtype=torch.float32)
    features = extract_features(model, train_inputs_tensor, model_name=args.model, mode=mode)
    scaled_features = scale_normal(features)
    # Remove empty (unused) feature columns
    non_empty_mask = scaled_features.abs().sum(dim=0).bool()
    scaled_features = scaled_features[:, non_empty_mask]
    
    robust_volumes = []
    volumes = []
    each_times = []

    for i in range(args.num_parties): # last iteration for grand coalition of datasets
        print('Evaluating the value of party {} ...'.format(i))

        # Get indices for data points leaving out party i
        extra_args = {
            'num_parties': args.num_parties,
            'given_indices': given_indices,
        }
        coalition_i_indices = get_indices_on_split(train_inputs, train_labels, 
                                                   party_i=i, split=args.split_method, 
                                                   extra_args=extra_args, loo=args.loo)
        
        coalition_i_inputs_tensor = scaled_features[coalition_i_indices]
                
        each_start_time = time.time()
        volume = compute_volume(coalition_i_inputs_tensor)
        X_tilde, cubes = compute_X_tilde_and_counts(coalition_i_inputs_tensor, omega=0.1)
        robust_vol = compute_robust_volume(X_tilde, cubes)
        
        # print('Volume:', volume)
        print('RV:', robust_vol)
        volumes.append(volume.item())
        robust_volumes.append(robust_vol.item())
        each_times.append(time.time() - each_start_time)
    total_time = time.time() - start_time
    
    np.savez('results/robust_volume_{}_{}_{}_{}_kept{}_seed{}.npz'.format(args.dataset, args.model, args.num_parties, args.split_method, args.trim_dataset, args.seed),
             volumes=volumes,
             robust_volumes=robust_volumes,
             total_time=total_time,
             each_times=each_times)
        
        
if __name__ == '__main__':
    main()

    