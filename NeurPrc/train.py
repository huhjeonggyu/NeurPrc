from NeurPrc.dataset import Dataset
from NeurPrc.net import Model
from NeurPrc.init import ATTR
from NeurPrc.util import load_best_model_and_optimizer, save_best_model_and_optimizer, read_finish_signal
import numpy as np
import torch
import datetime
import time
import os
import sys

def train(net) :

    os.makedirs("net",  exist_ok=True)

    ######################################### NeurPrc_Net
    save_dir = ATTR.save_dir
    n_path = net.n_path

    node_num_per_layer = net.node_num_per_layer
    hidden_layer_num = net.hidden_layer_num
    act_fn_type = net.act_fn_type
    fin_act_fn_type = net.fin_act_fn_type

    lr = net.learning_rate
    batch_size = net.batch_size 
    eval_batch_size = net.eval_batch_size 
    chunk_size = net.chunk_size 
    eval_chunk_size = net.eval_chunk_size 
    num_workers = net.num_workers

    print_freq = net.print_freq
    time_limit = net.time_limit
    test_freq = net.test_freq
    best_time = net.best_time
    continue_mode = net.continue_mode
    ######################################### 

    model = Model(node_num_per_layer,hidden_layer_num,act_fn_type,fin_act_fn_type)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    best_file_name,best_errors,total_time_pre,test_file_num = load_best_model_and_optimizer(model,opt,best_time,lr,continue_mode)
    print("*"*125)

    model = model.cuda()
    if ATTR.tdtype == torch.float  : model = model.float()
    if ATTR.tdtype == torch.double : model = model.double()
    print(model)

    if   continue_mode : mode = "at"
    else               : mode = "wt"
    file_ = open("eloss",mode)
    print("start",file=file_,flush=True)

    total_start = time.time()

    init = True
    test_cost_minute = 10000
    while True :

        if test_cost_minute >= test_freq :

            while True :

                dataset_train      = Dataset(save_dir["high"],chunk_size)
                dataset_train_eval = Dataset(save_dir["high"],eval_chunk_size)
                dataset_test1      = Dataset(save_dir["mid"], eval_chunk_size)
                dataset_test2      = Dataset(save_dir["low"], eval_chunk_size)

                print("*"*125)

                print("# of train data files: %d"%dataset_train.file_num)
                print("# of test1 data files: %d"%dataset_test1.file_num)
                print("# of test2 data files: %d"%dataset_test2.file_num)

                if (dataset_test1.file_num > 1.2*test_file_num[0]) and (dataset_test2.file_num > 1.2*test_file_num[1]) : 
                    reset = True
                    print("reset: True")
                    test_file_num = [dataset_test1.file_num, dataset_test2.file_num]
                else :
                    reset = False
                    print("reset: False")
                print("*"*125)

                if (dataset_train.file_num > 0) and (dataset_test1.file_num > 0) and (dataset_test2.file_num > 0) :
                    test_start = time.time()
                    break
                else :
                    time.sleep(10)

                if read_finish_signal() : 
                    file_.close()
                    sys.exit()

        start = time.time()

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, persistent_workers=True)
        dataloader_train_eval = torch.utils.data.DataLoader(dataset_train_eval, batch_size=eval_batch_size, num_workers=num_workers)
        dataloader_test1 = torch.utils.data.DataLoader(dataset_test1, batch_size=eval_batch_size, num_workers=num_workers)
        dataloader_test2 = torch.utils.data.DataLoader(dataset_test2, batch_size=eval_batch_size, num_workers=num_workers)

        for iters,batch in enumerate(dataloader_train) :

            if iters%print_freq == 0 : 
                loss_ = []

            opt.zero_grad()

            x_batch,y_batch = batch
            x_batch = x_batch.cuda(non_blocking=True).reshape(-1,x_batch.shape[-1])
            y_batch = y_batch.cuda(non_blocking=True).reshape(-1,1)

            y_pred_batch = model(x_batch)

            loss = torch.mean( (y_batch-y_pred_batch)**2 )
            loss_.append(loss.item())

            loss.backward()
            opt.step()

            if read_finish_signal() : 
                file_.close()
                sys.exit()

            if iters%print_freq == print_freq-1 or iters == len(dataloader_train)-1 :
                loss_ = np.mean(loss_)
                print("%7.2e"%lr,"%6d"%(iters+1),"%6d"%len(dataloader_train),"loss: %13.6e"%loss_.item(),flush=True)

        last = time.time()
        cost_time = last-start
        print("%6.3f(s) took"%cost_time)
        print("*"*125)

        test_last = time.time()
        test_cost_time = (test_last-test_start)
        test_cost_minute = test_cost_time//60

        if test_cost_minute >= test_freq :

            start = time.time()

            with torch.no_grad() :
                
                def test_loop(dataloader) :
                    eloss = []
                    for iters,batch in enumerate(dataloader) :

                        x_batch,y_batch = batch
                        x_batch = x_batch.cuda(non_blocking=True).reshape(-1,x_batch.shape[-1])
                        y_batch = y_batch.cuda(non_blocking=True).reshape(-1,1)

                        y_pred_batch = model(x_batch)

                        loss = torch.mean( (y_batch-y_pred_batch)**2 )
                        eloss.append(loss.item())

                        if read_finish_signal() : 
                            file_.close()
                            sys.exit()

                    return np.mean(eloss)

                print("training set is evaluated")
                error_train = test_loop(dataloader_train_eval)
                print("test1 set is evaluated")
                error_test1 = test_loop(dataloader_test1)
                print("test2 set is evaluated")
                error_test2 = test_loop(dataloader_test2)

                error_net = (10*error_test1-error_test2)/9
                error_app = error_train - error_net
                N_net = error_app/error_net * n_path

            total_last = time.time()
            total_time = total_last-total_start
            total_time += total_time_pre

            tmp = total_time
            hour = tmp//3600
            tmp = total_time - (3600*hour)
            minute = tmp//60
            second = tmp%60
            print("(total) %4d(h) %4d(m) %6.2f(s) took (time_limit: %d)"%(hour,minute,second,time_limit))
            print("*"*125,flush=True)

            now = datetime.datetime.now()
            now = '%s_%s_%s_%s_%s_%s'%(now.year,now.month,now.day,now.hour,now.minute,now.second)

            errors = [error_train,error_test1,error_test2]
            best_file_name,best_errors,marks = save_best_model_and_optimizer(
                        model,opt,now,errors,best_errors,best_file_name,total_time,test_file_num,init,reset)
            mark1,mark2 = marks

            last = time.time()
            cost_time = last-start

            print("*"*125)
            print("%-18s"%now,"%7.2e >"%lr,"%13.6e"%error_train,"(%13.6e,%13.6e) ="%(error_test1,error_test2),
                    "%13.6e +"%error_app,"%13.6e"%error_net,"> %13.6e"%N_net,"> %1s"%mark1,"%1s"%mark2)
            print("%-18s"%now,"%7.2e >"%lr,"%13.6e"%error_train,"(%13.6e,%13.6e) ="%(error_test1,error_test2),
                    "%13.6e +"%error_app,"%13.6e"%error_net,"> %13.6e"%N_net,"> %1s"%mark1,"%1s"%mark2,file=file_,flush=True)
            print("%6.3f(s) took"%cost_time)
            print("*"*125,flush=True)

            hour = total_time//3600
            if hour >= time_limit : break

        init = False

    file_.close()
