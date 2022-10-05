import torch
import numpy as np
import glob
import os

#################################################################################

def read_finish_signal() :
    with open('finish','rt') as f :
        sig = int(f.read().strip())
        if sig == 1 : return True
        else        : return False

def make_dataset_impl(args) :
    dat = []
    for arg in args :
        if   np.isscalar(arg)             : pass
        elif isinstance(arg,torch.Tensor) : arg = arg.cpu().numpy()
        else                              : arg = arg.data.cpu().numpy()
        dat.append(arg)
    return np.vstack(dat).T

def make_dataset(xargs,yargs) :
    xdat = make_dataset_impl(xargs)
    ydat = make_dataset_impl(yargs)
    return (xdat,ydat)

from scipy.stats import norm
def bs(x) :
    T = x[:,1:2]
    vol = x[:,2:3]
    lnK = x[:,0:1]*np.sqrt(T)
    K = np.exp(lnK)
    d1 = (-lnK+0.5*vol**2*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return norm.cdf(d1) - K*norm.cdf(d2)

def test(rank,size) :
    mc = BS_Exact_MC()
    ATTR.init(mc,rank,size)

    n_path = int(1e3)
    x_dat,y_dat = mc(n_path)
    t_dat = bs(x_dat)
    err = y_dat-t_dat
    print(x_dat[:10])
    print(y_dat[:10])
    print(t_dat[:10])
    print(err[:10])
    print(x_dat.shape)
    print(y_dat.shape)
    tmp = np.concatenate([x_dat,y_dat],axis=1)
    print(tmp.shape)

#################################################################################

def load_best_model_and_optimizer(model,opt,best_time,lr,continue_mode) :

    best_model_name = glob.glob('net/best_model_%s.pkl'%best_time)
    if continue_mode : best_model_name = glob.glob('net/model.pkl')

    if best_model_name :

        best_model_name = best_model_name[0]
        state_dict = torch.load(best_model_name,map_location="cpu")

        model_param_names = [e[0] for e in model.named_parameters()]
        state_dict_keys = [e for e in state_dict.keys()]
        flag = True
        for e1 in model_param_names :
            for i,e2 in enumerate(state_dict_keys) :
                if e1 == e2 : break
                if i == len(state_dict_keys)-1 : flag = False
            if not flag : break

        if flag : 
            model.load_state_dict(torch.load(best_model_name,map_location="cpu"))
        else :
            #['layer_i.weight', 'layer_i.bias', 'layers_h.0.weight', 'layers_h.0.bias', 'layers_h.1.weight', 'layers_h.1.bias', 'layer_o.weight', 'layer_o.bias']
            #['0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias', '6.weight', '6.bias']
            model.layer_i.weight.data     = state_dict['0.weight']
            model.layer_i.bias.data       = state_dict['0.bias']
            model.layers_h[0].weight.data = state_dict['2.weight']
            model.layers_h[0].bias.data   = state_dict['2.bias']
            model.layers_h[1].weight.data = state_dict['4.weight']
            model.layers_h[1].bias.data   = state_dict['4.bias']
            model.layer_o.weight.data     = state_dict['6.weight']
            model.layer_o.bias.data       = state_dict['6.bias']
        print('%s has been loaded'%best_model_name)

    else :
        best_model_name = ''

    best_opt_name = glob.glob('net/best_opt_%s.pkl'%best_time)
    if continue_mode : best_opt_name = glob.glob('net/opt.pkl')

    if best_opt_name :
        best_opt_name = best_opt_name[0]
        opt.load_state_dict(torch.load(best_opt_name,map_location="cpu"))
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print('%s has been loaded'%best_opt_name)
    else :
        best_opt_name = ''

    for g in opt.param_groups:
        g['lr'] = lr

    best_err_name = glob.glob('net/best_err_%s.pkl'%best_time)
    if continue_mode : best_err_name = glob.glob('net/err.pkl')

    if best_err_name :
        best_err_name = best_err_name[0]
        best_errors = torch.load(best_err_name,map_location="cpu")
        print(best_errors)
        print('%s has been loaded'%best_err_name)
    else :
        best_err_name = ''
        best_errors = [1.,1.,1.]
        
    best_file_name = [best_model_name,best_opt_name,best_err_name]

    if continue_mode : time_file_name = glob.glob('net/total_time.pkl')
    else             : time_file_name = []

    if time_file_name :
        time_file_name = time_file_name[0]
        total_time = torch.load(time_file_name,map_location="cpu")
        print(total_time)
        print('%s has been loaded'%time_file_name)
    else : 
        total_time = 0.

    if continue_mode : test_file_name = glob.glob('net/test_file_num.pkl')
    else             : test_file_name = []

    if test_file_name :
        test_file_name = test_file_name[0]
        test_file_num = torch.load(test_file_name,map_location="cpu")
        print(test_file_num)
        print('%s has been loaded'%test_file_name)
    else : 
        test_file_num = [0,0]

    return (best_file_name,best_errors,total_time,test_file_num)

def save_best_model_and_optimizer(model,opt,now,errors,best_errors,best_file_name,total_time,test_file_num,init,reset) :

    torch.save(model.state_dict(),'net/model.pkl')
    torch.save(opt.state_dict(),'net/opt.pkl')
    torch.save(errors,'net/err.pkl')
    torch.save(test_file_num,'net/test_file_num.pkl')
    print('model,optimizer and error have been saved')

    error_train,error_test1,error_test2 = errors
    best_error_train,best_error_test1,best_error_test2 = best_errors

    if  best_error_train > error_train : 
        best_error_train = error_train 
        mark1 = 'o'
    else :                               
        mark1 = ' '

    if ((best_error_test1 > error_test1) and (best_error_test2 > error_test2)) or reset : 

        best_model_name,best_opt_name,best_err_name = best_file_name

        best_model_name_pre = best_model_name
        best_model_name ='net/best_model_%s.pkl'%now
        torch.save(model.state_dict(),best_model_name)

        best_opt_name_pre = best_opt_name
        best_opt_name = 'net/best_opt_%s.pkl'%now
        torch.save(opt.state_dict(),best_opt_name)

        best_error_test1 = error_test1
        best_error_test2 = error_test2
        mark2 = 'o'

        best_err_name_pre = best_err_name
        best_err_name = 'net/best_err_%s.pkl'%now
        torch.save([error_train,error_test1,error_test2],best_err_name)
        print('best model,optimizer and error have been saved: %s'%now)

        if not init :
            if os.path.exists(best_model_name_pre) and (best_model_name_pre.find('best')!=-1) : 
                os.remove(best_model_name_pre)
            if os.path.exists(best_opt_name_pre)   and (best_opt_name_pre.find('best')  !=-1) :   
                os.remove(best_opt_name_pre)
            if os.path.exists(best_err_name_pre)   and (best_err_name_pre.find('best')  !=-1) :   
                os.remove(best_err_name_pre)

        best_file_name = [best_model_name,best_opt_name,best_err_name]

    else :
        mark2 = ' '

    torch.save(total_time,'net/total_time.pkl')

    best_errors    = [best_error_train,best_error_test1,best_error_test2]
    marks          = [mark1,mark2]
    return (best_file_name,best_errors,marks)
