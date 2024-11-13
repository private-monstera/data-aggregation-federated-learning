import os
import copy
import time
import numpy as np
from tqdm import tqdm
import quantizer as qn

import torch
from tensorboardX import SummaryWriter

from update import LocalUpdate, Evaluation
from models import CNNMnist
from cifar_model import CNNCifar, ResNet
from utils import get_dataset, average_weights, exp_details, get_quantization_bit
from options import args_parser
import prettytable as pt
def make_dir(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)
def train():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    args = args_parser()
    exp_details(args)
    device = 'cuda'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Build model for different dataset
    if args.dataset == 'mnist':
        quant = lambda: qn.BlockQuantizer(args.quant_bits, args.quant_bits, args.quant_type)
        global_model = CNNMnist(args=args, quant=quant)
    elif args.dataset == 'cifar':
        quant = lambda: qn.BlockQuantizer(args.quant_bits, args.quant_bits, args.quant_type)
        quantx = lambda x: qn.BlockQuantizer(x, args.quant_bits, args.quant_bits, args.quant_type)
        if args.iid==1:
            global_model = ResNet(args=args, quant=quant, quantx=quantx)
        else:
            global_model = CNNCifar(args=args, quant=quant)
    # hold global weight get from server
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    # Record training loss and accuracy
    train_loss, train_accuracy = [], []
    print_every = 1
    acc_level = np.array(list(range(41)))*0.01+0.6
    acc_true=[]
    acc_table_line=[]
    acc_flag=np.zeros_like(acc_level)
    quant_bit_for_user, avg_for_user = get_quantization_bit(args)
    last_max_acc=0
    # Set the filename for saving results
    result_base_filename = 'result/' + args.dataset + '/iid/' + args.average_scheme
    if args.iid==0:
        result_base_filename = 'result/' + args.dataset + '/noniid/' + args.average_scheme
    save_acc_filename=result_base_filename + '/c'+str(int(args.frac*10))+'result_04-'+str(int(args.bit_4_ratio*10))+'_8-'+str(int(args.bit_8_ratio*10))+'.txt'
    save_pkl_filename = result_base_filename +'/c'+str(int(args.frac*10))+'result_04-' + str(int(args.bit_4_ratio * 10)) + '_8-' + str(int(args.bit_8_ratio * 10)) + '.pkl'

    make_dir(result_base_filename)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        cur_q=[]
        for idx in idxs_users:
            quantx=None
            quant = None
            idx = int(idx)
            # Set the quantization condition and local model for each client
            if quant_bit_for_user[idx]!=0:
                args.quant_bits = quant_bit_for_user[idx]
                quant = lambda: qn.BlockQuantizer(args.quant_bits, args.quant_bits, args.quant_type)
                quantx = lambda x: qn.quantize_block(x, args.quant_bits, args.quant_bits, args.quant_type)
            if args.dataset == 'mnist':
                user_model = CNNMnist(args=args, quant=quant)
            elif args.dataset == 'cifar' and args.iid==1:
                user_model = ResNet(args=args, quant=quant, quantx=quantx)
            elif args.dataset == 'cifar' and args.iid==0:
                user_model = CNNCifar(args=args, quant=quant)
            user_model.to(device)
            user_model.train()
            # Update the weights of local to center weights
            weight_name = []
            for i in global_weights:
                weight_name.append(i)
            cnt = 0
            user_weights = user_model.state_dict()
            for i in user_weights:
                if quantx!=None:
                    user_weights[i] = quantx(global_weights[weight_name[cnt]].to(float))
                else:
                    user_weights[i] = global_weights[weight_name[cnt]]
                cnt += 1
            user_model.load_state_dict(user_weights)
            # Train the local model
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger,quant=quantx, quantbit=args.quant_bits, mode=args.quant_type)
            w, loss, q = local_model.update_weights(model=copy.deepcopy(user_model))
            cur_q.append(q)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            torch.cuda.empty_cache()

        # update global weights
        global_weights = average_weights(args, local_weights,avg_for_user[idxs_users],q_for_user=np.array(cur_q))
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all clients at every epoch
        list_acc, list_loss = [], []
        weight_name = []
        for i in global_weights:
            weight_name.append(i)
        for c in range(args.num_users):
            quant = None
            quantx = None
            # Set the quantization condition and local model for each client
            if quant_bit_for_user[c] != 0:
                args.quant_bits = quant_bit_for_user[c]
                quant = lambda: qn.BlockQuantizer(args.quant_bits, args.quant_bits, args.quant_type)
                quantx = lambda x: qn.quantize_block(x, args.quant_bits, args.quant_bits, args.quant_type)
            if args.dataset == 'mnist':
                user_model = CNNMnist(args=args, quant=quant)
            elif args.dataset == 'cifar' and args.iid == 1:
                user_model = ResNet(args=args, quant=quant, quantx=quantx)
            elif args.dataset == 'cifar' and args.iid == 0:
                user_model = CNNCifar(args=args, quant=quant)
            user_model.to(device)
            user_model.eval()
            cnt = 0
            user_weights = user_model.state_dict()
            for i in user_weights:
                user_weights[i] = global_weights[weight_name[cnt]]
                cnt += 1
            user_model.load_state_dict(user_weights)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger, quant=quantx, quantbit=quant_bit_for_user[c],mode=args.quant_type)
            acc, loss = local_model.inference(model=user_model)
            list_acc.append(acc)
            list_loss.append(loss)
            torch.cuda.empty_cache()
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args, quant=None)
        elif args.dataset == 'cifar' and args.iid == 1:
            global_model = ResNet(args=args, quant=quant, quantx=quantx)
        elif args.dataset == 'cifar' and args.iid == 0:
            global_model = CNNCifar(args=args, quant=quant)
        cnt = 0
        current_weight = global_model.state_dict()
        for i in current_weight:
            current_weight[i] = global_weights[weight_name[cnt]]
            cnt += 1
        global_model.load_state_dict(current_weight)
        global_model.to(device)
        global_model.eval()
        test_acc, test_loss = Evaluation(args, global_model, test_dataset)
        # Save the center model with the maximum accuracy
        if test_acc>last_max_acc:
            last_max_acc=test_acc
            torch.save(global_model.state_dict(), save_pkl_filename)
        with open(save_acc_filename,'a') as file:
            write_content=str(epoch+1)+' '+str(np.mean(np.array(train_loss)))+' '+str(train_accuracy[-1])+' '+str(test_acc)+"\n"
            file.write(write_content)
        for acc_index in range(len(acc_level)):
            if acc_flag[acc_index]==0 and test_acc>=acc_level[acc_index] and test_acc not in acc_true and test_acc>=acc_level[acc_index+1]:
                acc_flag[acc_index]=1
            if acc_flag[acc_index]==0 and test_acc>=acc_level[acc_index] and test_acc not in acc_true:
                acc_true.append(test_acc)
                acc_flag[acc_index]=1
                acc_table_line.append(epoch+1)
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}%'.format(100*train_accuracy[-1]))
            print("Test  Accuracy: {:.2f}%\n".format(100 * test_acc))
            print(f"Test  Loss: {np.mean(np.array(test_loss))}")
            print('4: ',args.bit_4_ratio,', 8: ',args.bit_8_ratio,', ',args.average_scheme)
            if len(acc_table_line) == 0:
                print("cannot get the communication round for the target accuracy")
            else:
                table = pt.PrettyTable()
                table.field_names = acc_true  # acc_level
                table.add_row(acc_table_line)
                print(table)
        if (epoch+1)%10==0 and args.lr>=1e-4:
            args.lr=args.lr*0.9
        print('learning rate : ',args.lr)
    # Test inference after completion of training
    test_acc, test_loss = Evaluation(args, global_model, test_dataset)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    table = pt.PrettyTable()
    table.field_names = acc_true
    table.add_row(acc_table_line)
    test_acc=0
    if len(acc_table_line) == 0:
        print("cannot get the communication round for the target accuracy")
    else:
        table = pt.PrettyTable()
        table.field_names = acc_true
        table.add_row(acc_table_line)
        print(table)
    return train_accuracy[-1],test_acc
