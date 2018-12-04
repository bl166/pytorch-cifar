from __future__ import print_function

import torch, torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models, transforms, datasets
import numpy as np

import os, sys, time, requests, copy
from torchsummary import summary
from bashplotlib.histogram import plot_hist
import matplotlib.pyplot as plt

from utils import progress_bar


# ----- Set GPU and random seed -----
seed = 0
torch.manual_seed(seed)

use_gpu = 0
if use_gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    torch.cuda.set_device(use_gpu)
    use_gpu = torch.cuda.is_available()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    use_gpu = False
    
device = 'cuda' if use_gpu else 'cpu'

if use_gpu:
    torch.cuda.manual_seed_all(seed)
    
# ------------------------------------


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def count_parameters(m):
    return np.sum(p.numel() for p in m.parameters() if p.requires_grad)



def replace_layers(model, i, indices, layers):
    if i in indices:
        return layers[indices.index(i)]
    return model[i]


def initiate_pruned_layer(conv, length, type):
    return nn.Conv2d(in_channels = conv.in_channels - (type == 'next')*length, \
                out_channels = conv.out_channels - (type == 'curr')*length,
                kernel_size = conv.kernel_size, \
                stride = conv.stride,
                padding = conv.padding,
                dilation = conv.dilation,
                groups = conv.groups,
                bias = (conv.bias is not None)).cuda()


def prune_excecute(old_data, filter_dim, multi_prune = 1):
    """
    :param: (tuple) filter_dim [0] (list, tuple, or int) which filters to remove
                               [1] (int) along which axes to remove
    """
    fd, fa = filter_dim
    if isinstance(fd, int):
        fd = [fd]

    fd_ = [range(d * multi_prune, (d + 1) * multi_prune) for d in fd]
    filter = np.s_[[i for j in fd_ for i in j]]

    new_data = np.delete(old_data.cpu().numpy(), filter, fa)
    return torch.from_numpy(new_data).cuda()


def prune_vgg_helper(model_in, layer_index, filter_indices):

    model = copy.deepcopy(model_in)
    
    # verify classifier
    if not isinstance(model.classifier,nn.Sequential):
        seq_cls = nn.Sequential(model.classifier)
        del model.classifier
        model.classifier = seq_cls

    """
    Excecute pruning on the layer#<layer_index> by removing filter#<filter_indices>
    and set previous and following layers to appropraite shapes

    :Param: (torchvision.models) model
    :Param: (int) layer_index
    :Param: (int or list) filter_indices
    """

    _, conv = list(model.features._modules.items())[layer_index]
    next_conv = None # next conv layer
    batch_norm = None   # next batch normalization
    offset = 1
    
    # initiate with layer index to prune
    replace_index_list = [layer_index] 
    
    # dont know the new weights and bias yet, leave blank
    replace_content_list = []

    # feature modules list 
    feature_items = list(model.features._modules.items())
    
    while layer_index + offset < len(feature_items):
        
        res = feature_items[layer_index + offset][1]
        
        if batch_norm is None and isinstance(res, nn.BatchNorm2d):
            batch_norm = res
            
            # add batch norm layer index to the list if exists
            replace_index_list.append(layer_index + offset)
            
        if isinstance(res, nn.modules.conv.Conv2d):
            next_conv = res
            
            # add next conv layer index to the list if exists
            replace_index_list.append(layer_index + offset)
            break
            
        offset += 1
        
    new_conv = initiate_pruned_layer(conv, len(filter_indices), 'curr')
    new_conv.weight.data = prune_excecute(conv.weight.data, (filter_indices, 0))
    new_conv.bias.data   = prune_excecute(conv.bias.data,   (filter_indices, 0))
        
    # add new conv layer content to the list
    replace_content_list.append(new_conv)
    
    if batch_norm:
        new_bn = nn.BatchNorm2d(batch_norm.num_features-len(filter_indices)).cuda()
        new_bn.weight.data = prune_excecute(batch_norm.weight.data, (filter_indices, 0))
        new_bn.bias.data   = prune_excecute(batch_norm.bias.data,   (filter_indices, 0))
                
        # add the following batch norm content to the list
        replace_content_list.append(new_bn)

    if next_conv:
        next_new_conv = initiate_pruned_layer(next_conv, len(filter_indices), 'next')
        next_new_conv.weight.data = prune_excecute(next_conv.weight.data, (filter_indices, 1))
        next_new_conv.bias.data   = next_conv.bias.data    
        
        # add new conv layer content to the list
        replace_content_list.append(next_new_conv)
    
    else:
        #Prunning the last conv layer. This affects the first linear layer of the classifier.
        classif_items = list(model.classifier._modules.items())
                    
        layer_index = 0
        old_linear_layer = None
        for _, module in classif_items:
            if isinstance(module, nn.Linear):
                old_linear_layer = module
                break
            layer_index += 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")

        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
             nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                 old_linear_layer.out_features)

        new_linear_layer.weight.data = prune_excecute(
                old_data = old_linear_layer.weight.data, # old weights
                filter_dim = (filter_indices, 1),
                multi_prune = params_per_input_channel
        )
        new_linear_layer.bias.data = old_linear_layer.bias.data

        classifier = nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        model.classifier = classifier

    # replace the layers in the model        
    features = nn.Sequential(
                *(replace_layers(model.features, i, replace_index_list, \
                    replace_content_list) for i, _ in enumerate(model.features)))
    del model.features
    model.features = features

    return model.cuda()



def get_prunning_plan(w_tensor, c_p_tuple, measure="abs_mean"):

    weights = w_tensor.data.cpu().numpy() # gpu method

    if measure=="random":
        to_prune = np.random.choice(*c_p_tuple, replace=False)

    elif measure=="abs_mean":
        channel_mean = np.mean(np.abs(weights), axis=(1,2,3))
        assert len(channel_mean)==c_p_tuple[0]
        to_prune = channel_mean.argsort()[:c_p_tuple[1]]

    elif measure=="mean":
        channel_mean = np.mean(weights, axis=(1,2,3))
        assert len(channel_mean)==c_p_tuple[0]
        to_prune = channel_mean.argsort()[:c_p_tuple[1]]
        
    elif measure=="bn":
        assert len(weights)==c_p_tuple[0]
        to_prune = weights.argsort()[:c_p_tuple[1]]
        
    else:
        raise NotImplementedError("Sparsity measurement not implemented!")

    return to_prune



def prune_conv_layer(model_in, sparsity, criterion):

    # find all conv layers and number of channels
    conv_layer_indices, num_channels = find_type_layers(model_in.features,"conv2d")
    if criterion=='bn':
        bn_layer_indices, num_channels_bn = find_type_layers(model_in.features,"batchnorm")
        assert [i+1 for i in conv_layer_indices]==bn_layer_indices and num_channels==num_channels_bn

    # find number of channels to prune for each layers
    num_prune = [np.ceil(c * sparsity).astype(int) for c in num_channels]

    # construct a look-up table for pruning
    prune_dict = dict(zip(conv_layer_indices, list(zip(num_channels,num_prune))))

    # parameters
    weights = list(model_in.features._modules.items())

    # prune from backwards
    for l in sorted(conv_layer_indices, reverse=True):
        chnl_to_prune = get_prunning_plan(weights[l+int(criterion=='bn')][1].weight, prune_dict[l], criterion)
        model_in = prune_vgg_helper(model_in, l, chnl_to_prune)

    return model_in.cuda()



def prune_conv_layer_legacy(model_in, sparsity):

    # find all conv layers and number of channels
    conv_layer_indices, num_channels = find_type_layers(model_in.features,"conv2d")

    # find number of channels to prune for each layers
    num_prune = [np.ceil(c * sparsity).astype(int) for c in num_channels]

    # construct a look-up table for pruning
    prune_dict = dict(zip(conv_layer_indices, list(zip(num_channels,num_prune))))

    # parameters
    weights = list(model_in.features._modules.items())

    # prune from backwards
    for l in sorted(conv_layer_indices, reverse=True):
        chnl_to_prune = get_prunning_plan(weights[l][1].weight, prune_dict[l], "abs_mean")
        model_in = prune_vgg_helper(model_in, l, chnl_to_prune)

    return model_in.cuda()



def find_type_layers(module, type="conv2d"):
    """
    :Return: [0] The indices of layers in the given module of the specified type
             [1] Number of channels available to prune for each layer
    """

    find_dict = {
        "conv2d": nn.modules.conv.Conv2d,
        "batchnorm":nn.BatchNorm2d,
        "linear": nn.Linear
    }
    assert type in find_dict.keys(), "Type error!"

    index_list,nchannel_list = [],[]

    for i,mod in enumerate(module._modules.items()):
        if isinstance(mod[1], find_dict[type]):
            index_list.append(i)
            try:
                nchannel_list.append(mod[1].out_channels)
            except:
                nchannel_list.append(mod[1].num_features)
            

    return index_list, nchannel_list



global_weight = []
def get_conv_weights_global(l):
    if isinstance(l, nn.modules.conv.Conv2d):
#         print(type(l))
        global_weight.append(l.weight.data)
    elif '_modules' in dir(l):
        for _, next_l in l._modules.items():
            get_conv_weights_global(next_l)
            
def vis_weights_hist(model_in, axis = None):
    global global_weight
    del global_weight[:]
    get_conv_weights_global(model_in)
    params = global_weight
    
#     params = [p.weight.data for n, p in model_in.features._modules.items() if isinstance(p, nn.modules.conv.Conv2d)]

    samples = np.concatenate([w.data.cpu().numpy().flatten() for w in params])
#     print(samples.shape)

    # visualization
    if is_interactive():
        plot_hist_notebook(samples, axis)
    else:
        plot_hist(samples, height=30, bincount=100, xlab=True, showSummary=True)
    
    
def plot_hist_notebook(samples, axis):
    
    if axis is not None:
        xlabel = axis.set_xlabel
        ylabel = axis.set_ylabel
        title = axis.set_title
        grid = axis.grid
        xlim = axis.set_xlim
        ylim = axis.set_ylim
        hist = axis.hist
    else:
        xlabel = plt.xlabel
        ylabel = plt.ylabel
        title = plt.title
        grid = plt.grid
        xlim = plt.xlim
        ylim = plt.ylim
        hist = plt.hist
            
    weights = np.ones_like(samples)/len(samples)
    n, bins, patches = hist(samples, 150, density=0, facecolor='green', alpha=0.75, weights=weights)
#     n, bins, patches = hist(samples, 150, density=0, facecolor='green', alpha=0.75)

    xlabel('Weight values')
    ylabel('Probability')
    title('Weights Histogram')
#     xlim(np.min(samples),np.max(samples))
    xlim(-.1,.1)
    ylim(np.min(n),1)#np.max(n))
    grid(True)

    if axis is None:
        plt.show()
        
    
    
    
    
    
    
def prune_by_sparsity(model_origin, sparsity = 0, num_ep = 50, ft_lrate = 1e-2, 
                      res_df = None, plot=True, ctri = 'bn', ret = False):        
     
    ft = finetuner(loss="xent", optimizer="sgd", lrate=ft_lrate, sparsity=sparsity)
    
    # original model accuracy
    sys.__stdout__.write("\n* original model accuracy *\n")
    ft.test(model_origin)
    ft.best_acc = 0 # original acc does not count
 
    model_pruned = prune_conv_layer(model_origin, sparsity, criterion=ctri)

    if plot:
        _, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5), dpi=130)
            
        vis_weights_hist(model_origin, ax[0])
        vis_weights_hist(model_pruned, ax[1])
    
    # resume model if can 
    model_path = 'checkpoint/%s/ckpt_ft_%f.t7'%(model_pruned.name, sparsity)
    if os.path.exists(model_path):
        checkpoint_ft = torch.load(model_path)
        model_pruned.load_state_dict(checkpoint_ft['net'])
        ft.best_acc = checkpoint_ft['acc']
        start_epoch = checkpoint_ft['epoch']
    else:
        start_epoch = 0
        
    # right-after-pruning accuracy
    sys.__stdout__.write("\n* right-after-pruning / resumed accuracy *\n")
    ft.test(model_pruned)
           
    
    inference_time = np.zeros(num_ep)
    for ep in range(start_epoch, num_ep):
        model_pruned = ft.train(model_pruned, ep)
        # fine-tuning accuracy
        acc, inference_time[ep] = ft.test(model_pruned, ep, save=model_path)
        if res_df is not None:
            res_df.at[sparsity, ep] = acc
        
    if res_df is not None:
        res_df.at[sparsity, 'maxAcc'] = ft.best_acc
        res_df.at[sparsity, 'modSize'] = os.path.getsize(model_path)
        res_df.at[sparsity, 'infTime'] = np.mean(inference_time)
        
       
    sys.__stdout__.write("\n********************************\n")

    if plot:            
        vis_weights_hist(model_pruned, ax[2])
        plt.show()
    
    return model_pruned, res_df


# ----------------------------------
# Fine-tune after pruning to restore accuracy

from tensorboardX import SummaryWriter

class finetuner(object):
    
    best_acc = 0
    
    trainloader = None
    testloader = None
    criterion = None
    optimizer = None
    
    def __init__(self, loss="xent", optimizer="sgd", lrate=1e-2, sparsity=0):
        
        print("==> Finetuning ..")
        self.sparsity = sparsity
        
        self.writer = SummaryWriter()

        # -----------------------
        # Data preparation
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=8)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        
        # -----------------------
        # loss and optimizer
        
        if loss.lower()=="xent":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplentedError
            
        if optimizer.lower()=="sgd":
            self.optmize = lambda x: torch.optim.SGD(x, lr=lrate, momentum=0.9, weight_decay=5e-4)
        else:
            raise NotImplentedError

    def train(self, model, epoch = 0):
        sys.__stdout__.write('\nEpoch: %d\n' % epoch)
        
        if device=='cuda' and 'module' not in model._modules.keys():
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
                
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer = self.optmize(model.parameters())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # -----------------------
            # progress bar
            old_stdout = sys.stdout
            sys.stdout = open('/dev/stdout', 'w')

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            sys.stdout = old_stdout
            # -----------------------
            
        return model


    def test(self, model, epoch = 0, save = None):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        time_all = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                t0 = time.time()
                outputs = model(inputs)
                t1 = time.time()
                time_all += t1-t0
                
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # -----------------------
                # progress bar
                old_stdout = sys.stdout
                sys.stdout = open('/dev/stdout', 'w')

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
                sys.stdout = old_stdout
                # -----------------------
                
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            self.best_acc = acc
            if save:# or epoch==0:
                sys.__stdout__.write('Saving..\n')
                if 'module' in model._modules.keys():
                    model = model.module
                    
                state = {
                    'net': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'cfg': model.cfg,
                }
#                 save_path = 'checkpoint/%s/ckpt_ft_%f.t7'%(model.name,self.sparsity)
                torch.save(state, save)
            
        return acc, time_all
        



        
        
        
        
        
if __name__ == "__main__":
    
    
    import torch.backends.cudnn as cudnn
    import os,sys
    sys.path.append(os.getcwd())
    from models import *  
    
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', '-m', default='vgg19', type=str, help='model architecture')
    args = parser.parse_args()
    
    timestamp = int(time.time())
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    net = VGG(args.model)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    ckpt_dir = './checkpoint/%s'%net.name
    if not os.path.exists(ckpt_dir):  
        os.makedirs(ckpt_dir)
    checkpoint_path = '%s/ckpt.t7'%ckpt_dir

    # if not already exists, train one
    if not os.path.exists(checkpoint_path):
        os.system("python main.py --resume --lr=0.01 --model %s"%net.name)
        
    net.load_state_dict(torch.load(checkpoint_path)['net'])

    print(net)
    
    
    import pandas as pd

    sparsity = np.linspace(0,1,21)[1:-1]#[0.36]#->0.64 #

    
    ft_ep = 100
    ft_leaningrate = 5e-2

    df_res = pd.DataFrame(np.zeros((len(sparsity), ft_ep+3)), index=sparsity)
    df_res.columns = list(df_res.columns)[:-3] + ['maxAcc', 'modSize', 'infTime']

    
    res_dir = './checkpoint/%s/results'%net.name
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    
    for s in sorted(sparsity, reverse=True):
        print("sparsity = %g"%s, end="\t")
        _, df_res = prune_by_sparsity(net, s, ft_ep, ft_leaningrate, df_res, plot=False)
        print("best accuracy = %g"%df_res.at[s, 'maxAcc'])
        df_res.to_pickle("%s/resDf_%d.pkl"%(res_dir,timestamp))

    print(df_res)
    
    
    
    
    
    
    
    
    
    
    
