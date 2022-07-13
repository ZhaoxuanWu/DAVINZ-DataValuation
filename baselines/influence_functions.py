"""
    Run python baselines/influence_functions.py from the project root director
    Example usage:
    
    python baselines/influence_functions.py --dataset=MNIST_baseline --model=ResNet18 --num_parties=10 --split_method=by_class --seed=0 --gpu=0
    python baselines/influence_functions.py --dataset=ising_baseline --model=CNN8 --num_parties=10 --split_method=random --seed=0 --gpu=0
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
########## Influence methods ##########
#######################################

def s_test(z_test, t_test, model, loss_fn, z_loader, gpu=-1, damp=0.01, scale=500.0,
           recursion_depth=20):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model, loss_fn, gpu)
    h_estimate = v.copy()

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        if i % 100 == 0:
            print('recursion ', i, '/', recursion_depth)
        #########################
        for x, t in z_loader:
            if gpu >= 0:
                x, t = x.to(device), t.to(device)
            y = model(x)
            loss = calc_loss(y, t, loss_fn)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            with torch.no_grad():
                h_estimate = [
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break # For stochastic estimation
    return h_estimate


def calc_loss(y, t, loss_fn):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)

    # y = torch.nn.functional.log_softmax(y)
    # loss = torch.nn.functional.nll_loss(
    #     y, t, weight=None, reduction='mean')
    
    loss = loss_fn(y, t)
    return loss


def grad_z(z, t, model, loss_fn, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    
    # Evaluate in epochs
    N = len(z)
    batch_size = 50 # Roughly
    n_batch = N // batch_size
    # initialize
    if gpu >= 0:
        z, t = z.to(device), t.to(device)
    
    params = [ p for p in model.parameters() if p.requires_grad ]    
    all_grads = None
    for i in range(n_batch):
        st = i * N // n_batch
        en = (i + 1) * N // n_batch
        y = model(z[st:en])
        curr_loss = calc_loss(y, t[st:en], loss_fn) * (en - st)
        g = list(grad(curr_loss, params))
        if all_grads == None:
            all_grads = g
        else:
            all_grads = [a + b for a, b in zip(all_grads, g)]
       
    all_grads = [a/N for a in all_grads]
    return all_grads


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    # first_grads = grad(y, w)
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    # return_grads = grad(elemwise_products, w, create_graph=True)
    return_grads = grad(elemwise_products, w)


    return return_grads


def calc_s_test(config, model, loss_fn, train_loader, test_loader, gpu):
    z_test, t_test = test_loader.dataset[:]
    r = config['r']
    scale = config['scale']
    recursion_depth = config['recursion_depth']
    all_s_test = s_test(z_test, t_test, model, loss_fn, train_loader,
                 gpu=gpu, scale=scale, recursion_depth=recursion_depth)
    
    for i in range(1, r):
        print('Calculating s_test repeat', i, '/', r)
        cur = s_test(z_test, t_test, model, loss_fn, train_loader,
               gpu=gpu, scale=scale, recursion_depth=recursion_depth)
        all_s_test = [a + c for a, c in zip(all_s_test, cur)]
    
    s_test_vec = [a / r for a in all_s_test]
    
    return s_test_vec

def main():
    given_indices = None
    cls_datasets = ['MNIST_baseline', 'MNIST_baseline_resized', 'CIFAR_10_baseline']
    reg_datasets = ['ising_baseline']
    
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
    optimizer = optim.SGD(model.parameters(), lr = lr if lr != None else 0.01) 
    
    start_time = time.time()
    # Train and save model
    train(model, loaders, loss_fn, optimizer, device, num_epochs=300, threshold=retrain_threshold, minimum_epochs=300)
    if mode == 'cls':
        accuracy, loss = test(model, loaders, loss_fn, device)
    else:
        loss = test_reg(model, loaders, loss_fn, device)
        accuracy = None
    print('Finished training the full model. Test Loss: {}; Test Acccuracy: {}'.format(loss, accuracy))
    save_path = 'checkpoints/{}_{}_influence_trained_model_seed{}.pt'.format(args.dataset.lower(), args.model.lower(), args.seed)
    torch.save(model, save_path)
    
    # Start calculating the influences
    config = {
        'r': 10, 
        'scale': 25.0, 
        'recursion_depth': 100,
        }
    
    method_start_time = time.time()
    s_test_vec = calc_s_test(config, model, loss_fn, loaders['train'], loaders['test'], gpu=int(args.gpu))
    s_test_vec = s_test_vec
    
    influences = []
    each_times = []
    for i in range(args.num_parties): # last iteration for grand coalition of datasets
        print('Evaluating the value of party {} ...'.format(i))

        each_start_time = time.time()
        # Get indices for data points leaving out party i
        extra_args = {
            'num_parties': args.num_parties,
            'given_indices': given_indices,
        }
        coalition_i_indices = get_indices_on_split(train_inputs, train_labels, 
                                                   party_i=i, split=args.split_method, 
                                                   extra_args=extra_args, loo=args.loo)
        
        coalition_i_inputs_tensor = torch.tensor(train_inputs[coalition_i_indices], dtype=torch.float32).reshape(-1, *dims)
        coalition_i_labels_tensor = torch.tensor(train_labels[coalition_i_indices])
        if mode == 'reg':
            coalition_i_labels_tensor = coalition_i_labels_tensor.float()
                    
        model = torch.load(save_path).to(device)
        grad_z_vec = grad_z(coalition_i_inputs_tensor, coalition_i_labels_tensor, model, loss_fn, gpu=int(args.gpu))

        s_test_flat = torch.cat([v.flatten() for v in s_test_vec])
        grad_z_flat = torch.cat([v.flatten() for v in grad_z_vec])
        # This influence is the LOO influence of the subset
        influence = torch.dot(s_test_flat, grad_z_flat)

        influences.append(influence.detach().cpu().numpy())
        each_times.append(time.time() - each_start_time)
    
    total_time = time.time() - start_time
    method_time = time.time() - method_start_time
    np.savez('results/influence_{}_{}_{}_{}_kept{}_seed{}.npz'.format(args.dataset, args.model, args.num_parties, args.split_method, args.trim_dataset, args.seed),
             influences=influences,
             each_times=each_times,
             total_time=total_time,
             method_time=method_time)


if __name__ == '__main__':
    main()
