"""
    Run python baselines/retrain.py from the project root directory
    Example usage:
    
    python baselines/retrain.py --dataset=MNIST_baseline --num_parties=10 --split_method=by_class --gpu=0 --seeds=5
    python baselines/retrain.py --dataset=CIFAR_10_baseline --num_parties=10 --split_method=by_class --gpu=0 --seeds=5
    python baselines/retrain.py --dataset=ising_baseline --num_parties=10 --split_method=random --gpu=0 --seeds=5
    
    python baselines/retrain.py --dataset=ising_quantity_aware --num_parties=10 --split_method=random --seeds=10 --gpu=0 --no-loo
    python baselines/retrain.py --dataset=ising_noise_stability --num_parties=10 --split_method=random --seeds=10 --gpu=0 --no-loo
    python baselines/retrain.py --dataset=MNIST_MNISTM --num_parties=10 --split_method=val_domain_shift --trim_dataset=10000 --seeds=10 --gpu=0 --no-loo
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
from utils import load_dataset_cls, load_dataset_reg, train, test, test_reg

from model.cnn import CNN
from model.resnet import BasicBlock, ResNet
from model.vgg import VGG
from model.cnn8 import CNN as CNN8
from model.mlp import MLP

parser = argparse.ArgumentParser(description='Data valuation at initialization.')
parser.add_argument('--gpu', help='gpu device index',
                    required=False,
                    type=str,
                    default='4')
parser.add_argument('--seeds', help='total number of seeds for reproducibility',
                    required=False,
                    type=int,
                    default=5)
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
        permutation_indices = np.random.RandomState(seed=0).permutation(size)
        each_party = size//extra_args['num_parties']
        if party_i == extra_args['num_parties']:
            party_i_indices = []
        else:
            party_i_indices = permutation_indices[np.arange(party_i * each_party, (party_i + 1) * each_party)]
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


def main():
    results = {}
    n_epochs = 300
    
    if args.dataset == 'ising_quantity_aware' or args.dataset == 'ising_noise_stability':
        model_list = ['MLP']
        lr = 0.1
    elif args.dataset == 'MNIST_MNISTM':
        model_list = ['CNN']
        lr = 0.01
    elif 'MNIST' in args.dataset or 'CIFAR' in args.dataset:
        model_list = ['ResNet18', 'VGG13']
        lr = 0.01
    elif 'ising' in args.dataset:
        model_list = ['CNN8', 'MLP']
        lr = 0.1
    else:
        raise NotImplementedError()
    
    for model_name in model_list:
        print(model_name)
        # Loading the dataset
        given_indices = None
        cls_datasets = ['MNIST_baseline', 'MNIST_baseline_resized', 'CIFAR_10_baseline', 'MNIST_MNISTM']
        reg_datasets = ['ising_baseline', 'ising_quantity_aware', 'ising_noise_stability']
                
        if args.dataset in cls_datasets:
            dataset_name = args.dataset
            if 'VGG' in model_name and dataset_name == 'MNIST_baseline':
                dataset_name = 'MNIST_baseline_resized'
            train_inputs, train_labels, test_inputs_tensor, test_labels, dims, num_to_keep = load_dataset_cls(dataset_name, args.trim_dataset, args.num_parties)
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
        
        curr_model_accuracies = []
        curr_model_losses = []
        curr_model_retrain_times = []
           
        for seed in range(args.seeds):
            print('Seed: ', seed)
            # Reproducibility
            set_seed(seed)
            
            # Construct NN model
            in_channels = dims[0]
            if model_name == 'CNN':
                if 'CIFAR_10' in args.dataset:
                    model = CNN(in_channels=in_channels, linear_dim=dims[2]//2//2).to(device)
                else:
                    model = CNN(in_channels=in_channels).to(device)
            elif model_name == 'ResNet18':
                model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels).to(device)
            elif model_name == 'ResNet21':
                model = ResNet(BasicBlock, [2, 3, 3, 3], in_channels=in_channels).to(device)
            elif model_name == 'ResNet34':
                model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels).to(device)
            elif model_name == 'VGG11':
                model = VGG('VGG11', in_channels=in_channels).to(device)
            elif model_name == 'VGG13':
                model = VGG('VGG13', in_channels=in_channels).to(device)
            elif model_name == 'VGG16':
                model = VGG('VGG16', in_channels=in_channels).to(device)
            elif model_name == 'MLP':
                model = MLP(in_dim=np.prod(dims), out_dim=1).to(device)
            elif model_name == 'CNN8':
                if len(dims) == 2:
                    dims = (1,) + dims
                model = CNN8(in_channels=1, out_dim=1, linear_dim=dims[1]//2//2).to(device)
            else:
                raise NotImplementedError()
            init_path = 'checkpoints/retrain_{}_{}_init_seed{}.pt'.format(args.dataset.lower(), model_name.lower(), seed)
            torch.save(model, init_path)
        
            accuracies = []
            losses = []
            retrain_times = []

            for i in range(args.num_parties + (1 if args.loo else 0)): # last iteration for grand coalition of datasets
                print('Evaluating the value of party {} ...'.format(i))

                # Get indices for data points leaving out party i
                extra_args = {
                    'num_parties': args.num_parties,
                    'given_indices': given_indices,
                }
                coalition_i_indices = get_indices_on_split(train_inputs, train_labels, 
                                                        party_i=i, split=args.split_method, 
                                                        extra_args=extra_args, loo=args.loo)
                coalition_i_inputs_tensor = torch.tensor(train_inputs[coalition_i_indices], dtype=torch.float32)
                coalition_i_labels_tensor = torch.tensor(train_labels[coalition_i_indices])

                t1 = time.time()
                if mode == 'cls':
                    coalition_i_data = TensorDataset(coalition_i_inputs_tensor.reshape(-1, *dims).to(device), coalition_i_labels_tensor.to(device))
                    test_data = TensorDataset(test_inputs_tensor.reshape(-1, *dims).float().to(device), torch.tensor(test_labels).to(device))
                    loss_func = torch.nn.CrossEntropyLoss()
                else:
                    coalition_i_data = TensorDataset(coalition_i_inputs_tensor.reshape(-1, *dims).to(device), coalition_i_labels_tensor.float().to(device))
                    test_data = TensorDataset(test_inputs_tensor.reshape(-1, *dims).float().to(device), torch.tensor(test_labels).float().to(device))
                    loss_func = torch.nn.MSELoss()
                
                loaders = {
                    'train' : torch.utils.data.DataLoader(coalition_i_data, 
                                                          batch_size=128, 
                                                          shuffle=True),

                    'test'  : torch.utils.data.DataLoader(test_data, 
                                                          batch_size=128, 
                                                          shuffle=False),
                }
                model = torch.load(init_path).to(device)
                optimizer = optim.SGD(model.parameters(), lr = lr) 
                
                train(model, loaders, loss_func, optimizer, device, num_epochs=n_epochs, minimum_epochs=n_epochs)
                
                if mode == 'cls':
                    accuracy, loss = test(model, loaders, loss_func, device)
                else:
                    loss = test_reg(model, loaders, loss_func, device)
                    accuracy = None
                losses.append(loss)
                accuracies.append(accuracy)
                print('Accuracy: ', accuracy)
                retrain_times.append(time.time() - t1)
                print('Retrain time: {}'.format(retrain_times[-1]))
            
            curr_model_losses.append(losses)
            curr_model_accuracies.append(accuracies)
            curr_model_retrain_times.append(retrain_times)
        results[model_name] = {'losses': curr_model_losses, 
                               'accuracies': curr_model_accuracies, 
                               'retrain_times': curr_model_retrain_times,}
                    
    np.savez('results/{}_retrain_{}_{}_kept{}_total_seeds{}.npz'.format(args.dataset, args.num_parties, args.split_method, args.trim_dataset, args.seeds),
             results=results)


if __name__ == '__main__':
    main()

