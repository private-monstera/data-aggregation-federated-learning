import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments fllowed from paper)
    parser.add_argument('--epochs', type=int, default=150,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of clie: n")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: K")
    parser.add_argument('--local_bs', type=int, default=600,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay of optimizer (default: 0.0005)')
    parser.add_argument('--average_scheme', type=str, default='FedHQ', help='choose average scheme',
                        choices=['FedAvg','Proportional','FedHQ'])
    parser.add_argument('--quant_bits', type=int, default=8, help='record the current quantization bit')
    parser.add_argument('--bit_4_ratio', type=float, default=0.6, help='the ratio for 4-bit clients')
    parser.add_argument('--bit_8_ratio', type=float, default=0.4, help='the ratio for 8-bit clients')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=1, help="To use CPU or GPU. Default set to use GPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--dir', type=str, default=None,
                        help='training directory (default: None)')
    parser.add_argument('--data_path', type=str, default="./data", required=False, metavar='PATH',
                        help='path to datasets location (default: "./data")')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='number of workers (default: 0)')
    parser.add_argument('--log_name', type=str, default='', metavar='S',
                        help="Name for the log dir")
    parser.add_argument('--quant_type', type=str, default='stochastic', metavar='S',
                        help='rounding method, stochastic or nearest ', choices=['stochastic', 'nearest'])

    args = parser.parse_args()
    return args
