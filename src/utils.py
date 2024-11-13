import copy
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from options import args_parser
import numpy as np
args = args_parser()

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
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
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                #raise NotImplementedError()
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def get_quantization_bit(args):
    quant_bit_for_user = np.zeros([args.num_users])
    avg_for_user = np.zeros([args.num_users])

    for user in range(args.num_users):
        if user < args.num_users * args.bit_4_ratio:
            quant_bit_for_user[user] = 4
            avg_for_user[user] = 4
        elif user < args.num_users * (args.bit_4_ratio + args.bit_8_ratio):
            quant_bit_for_user[user] = 8
            avg_for_user[user] = 8
        else:
            avg_for_user[user] = 64
    return quant_bit_for_user, avg_for_user

def average_weights(args,w,avg_for_user,q_for_user):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    user_num=len(avg_for_user)
    # Calculate the p for each user, the sum of p is 1
    p_for_user=np.ones(user_num)
    if args.average_scheme == 'FedAvg':
        p_for_user /= sum(p_for_user)
    if args.average_scheme == 'Proportional':
        p_for_user = np.array(avg_for_user) / np.sum(avg_for_user)
    if args.average_scheme == 'FedHQ':
        p_for_user = 1 / (1 + q_for_user)
        p_for_user /= np.sum(p_for_user)
    for key in w_avg:
        w_avg[key] *= p_for_user[0]
    for i in range(1, len(w)):
        weight_name = []
        for j in w[i]:
            weight_name.append(j)
        cnt=0
        for key in w_avg.keys():
            w_avg[key] += w[i][weight_name[cnt]] * p_for_user[i]
            cnt += 1
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset   : {args.dataset}')
    model='CNN'
    if args.dataset=='cifar' and args.iid==1:
        model='ResNet18'
    if args.dataset == 'cifar' and args.iid == 0:
        model = 'VGG11'

    print(f'    Model     : {model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
