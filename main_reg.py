"""
    Example usage:
    
    # Baselines
    python main_reg.py --dataset=ising_baseline --model=MLP --num_parties=10 --split_method=random --ground-truth --seed=0 --gpu=0
    python main_reg.py --dataset=ising_baseline --model=CNN8 --num_parties=10 --split_method=random --ground-truth --seed=0 --gpu=0

    # Awareness of data quantity
    python main_reg.py --dataset=ising_quantity_aware --model=MLP --num_parties=10 --split_method=random --ground-truth --seed=0 --gpu=0 --no-loo
    
    # Stability to noise
    python main_reg.py --dataset=ising_noise_stability --model=MLP --num_parties=10 --split_method=random --ground-truth --seed=0 --gpu=0 --no-loo
"""

import time
import os
import argparse
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from mmd import rbf_mmd2
from ntk import compute_ntk_score_batched, compute_ntk_score_batched_permute
from utils import load_dataset_reg, load_isling, load_isling_baseline, train, test, test_reg

from model.mlp import MLP
from model.cnn8 import CNN as CNN8

parser = argparse.ArgumentParser(description='Data valuation at initialization (regression task).')
parser.add_argument('--gpu', help='gpu device index',
                    required=False,
                    type=str,
                    default='0')
parser.add_argument('--seed', help='seed for reproducibility',
                    required=False,
                    type=int,
                    default=0)
parser.add_argument('--dataset', help='dataset to use: ising_baseline, ising_quantity_oriented, ising_noise_stability',
                    required=True,
                    type=str,
                    default='ising_baseline')
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
parser.add_argument('--model', help='model to use: CNN8, MLP',
                    required=True,
                    type=str,
                    default='GRU')
parser.add_argument('--loo', help='whether to compute leave-one-out value',
                    required=False,
                    type=bool,
                    default=True)
parser.add_argument('--no-loo', dest='loo', 
                    action='store_false')
parser.add_argument('--ground-truth', help='retrain models explicitly till convergence to get ground truth',
                    action='store_true')
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
    """
    Given the data, party index and split method name, return the indices of data samples concerned.
    
    Args:
        train_inputs (Tensor): the training inputs
        train_labels (Tensor): corresponding labels of the training inputs
        party_i (int): party index that is concerned
        split (str): a str specifying the type of split
        extra_args (dict): a dictionary containing extra args required for some specific split types
        loo (bool): if True, return the data sample indices excluding the data from party i
    Return:
        (np.array) the indices of data sample from a party i (or the complement if loo==True)
    """
    size = train_labels.shape[0]
    if split == 'random':
        cumsum = np.concatenate([[0], extra_args['dataset_sizes_cumsum']])
        if party_i != extra_args['num_parties']:
            party_i_indices = np.arange(cumsum[party_i], cumsum[party_i + 1])
        else:
            party_i_indices = []
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
    # Load the datasets
    train_inputs, train_labels, test_inputs_tensor, test_labels, dims, num_to_keep, dataset_sizes_cumsum = load_dataset_reg(args.dataset, args.split_method)
    args.trim_dataset = num_to_keep
    mode = 'reg'
            
    # Reproducibility
    set_seed(args.seed)
    
    # Construct NN model
    lr = 0.1
    diagonal_I_mag = 1e-4
    
    if args.model == 'MLP':
        input_dim = np.prod(dims)
        model = MLP(in_dim=input_dim, out_dim=1).to(device)
        retrain_threshold = 1e-10
        ntk_n_batch = 1
        use_hack = True
    elif args.model == 'CNN8':
        dims = (1,) + dims
        model = CNN8(in_channels=1, out_dim=1, linear_dim=dims[1]//2//2).to(device)
        retrain_threshold = 1e-8
        ntk_n_batch = 15
        use_hack = True
    else:
        raise NotImplementedError()

    init_path = 'checkpoints/{}_{}_init_seed{}.pt'.format(args.dataset.lower(), args.model.lower(), args.seed)
    torch.save(model, init_path)
        
    MMDs = []
    ntk_scores = []
    combined_scores = []
    accuracies = []
    losses = []
    min_eigens = []
    davinz_times = []
    retrain_times = []
    
    # DaVinz calculations

    for i in range(args.num_parties + (1 if args.loo else 0)): # Last iteration for grand coalition of datasets in LOO
        
        print('Evaluating the value of party {} ...'.format(i))

        # Get indices for data points leaving out party i
        extra_args = {
            'num_parties': args.num_parties,
            'dataset_sizes_cumsum': dataset_sizes_cumsum
        }
        coalition_i_indices = get_indices_on_split(train_inputs, train_labels, 
                                                   party_i=i, split=args.split_method, 
                                                   extra_args=extra_args, loo=args.loo)
        coalition_i_inputs_tensor = torch.tensor(train_inputs[coalition_i_indices], dtype=torch.float32)
        coalition_i_labels_tensor = torch.tensor(train_labels[coalition_i_indices], dtype=torch.float32)
        davinz_start_time = time.time()
        
        # MMD (out-of-domain generalization error)
        mmd_squared = rbf_mmd2(coalition_i_inputs_tensor.reshape(-1, np.prod(dims)), 
                               test_inputs_tensor.reshape(-1, np.prod(dims)).float(), 
                               sigma=5)
        mmd = torch.sqrt(mmd_squared)
        MMDs.append(mmd)
        print('MMD: {}'.format(mmd))

        # NTK (in-domain generalization error)
        model = torch.load(init_path).to(device)
        inputs = coalition_i_inputs_tensor.reshape(-1, *dims)
        coalition_i_labels_tensor = coalition_i_labels_tensor.reshape(-1)
        score, min_eigen = compute_ntk_score_batched_permute(model, inputs.to(device), 
                                                             coalition_i_labels_tensor.to(device),
                                                             mode, n_batch=ntk_n_batch, n_permute=1, use_hack=use_hack, 
                                                             diagonal_I_mag=diagonal_I_mag)
        ntk_scores.append(score)
        min_eigens.append(min_eigen)
        print('NTK: {}'.format(score))
        
        davinz_times.append(time.time() - davinz_start_time)
        print('DaVinz time: {}'.format(davinz_times[-1]))
                
        # Get the ground truth by retaining till convergence
        if args.ground_truth:
            retrain_start_time = time.time()
            coalition_i_data = TensorDataset(inputs.to(device), coalition_i_labels_tensor.reshape(-1, 1).to(device))
            test_data = TensorDataset(test_inputs_tensor.reshape(-1, *dims).float().to(device), torch.tensor(test_labels).float().to(device))
            
            loaders = {
                'train' : torch.utils.data.DataLoader(coalition_i_data, 
                                                      batch_size=128, 
                                                      shuffle=True),

                'test'  : torch.utils.data.DataLoader(test_data, 
                                                      batch_size=128, 
                                                      shuffle=False),
            }
            model = torch.load(init_path).to(device)
            loss_func = torch.nn.MSELoss()   
            optimizer = optim.SGD(model.parameters(), lr = lr) 
            
            train(model, loaders, loss_func, optimizer, device, num_epochs=3000, threshold=retrain_threshold, minimum_epochs=1000)
            loss = test_reg(model, loaders, loss_func, device)
            losses.append(loss)
            print('Loss: ', loss)
            retrain_times.append(time.time() - retrain_start_time)
            print('Retrain time: {}'.format(retrain_times[-1]))
    
    # Combining the in-domain and out-of-domain scores (aka v(S))
    # Note: a better alternative is to average across different runs/seeds - Eqn (4) of paper
    kappa = np.mean(MMDs)/ np.mean(ntk_scores)
    combined_scores = - kappa * np.array(ntk_scores) - np.array(MMDs)
    print('Combined scores: ', combined_scores)
    
    np.savez('results/{}_{}_{}_{}_kept{}_seed{}{}.npz'.format(args.dataset, args.model, args.num_parties, args.split_method, args.trim_dataset, args.seed, '_retrained' if args.ground_truth else ''),
             MMDs=MMDs,
             ntk_scores=ntk_scores,
             combined_scores=combined_scores,
             accuracies=accuracies,
             losses=losses,
             min_eigens=min_eigens,
             davinz_times=davinz_times,
             retrain_times=retrain_times)


if __name__ == '__main__':
    main()
