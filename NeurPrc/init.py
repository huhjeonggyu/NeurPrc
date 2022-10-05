import torch
import os

class Attr :

    def init(self,mc,makedir,rank,size) :

        if rank == 0 :
            with open("finish","wt") as f : 
                f.write("0")

        self.save_dir = {}
        for acc in ["high","mid","low"] :
            self.save_dir[acc] = "_".join([mc.model_name,mc.method_name,acc,mc.dtype,"gpu%d"%rank])
            if makedir :
                os.makedirs("data", exist_ok=True)
                os.makedirs(os.path.join("data",self.save_dir[acc]), exist_ok=True)

        self.device = "cuda:%d"%rank 

        ######################################### NeurPrc_MC
        self.prod_size = mc.prod_size 
        dtype = mc.dtype 
        self.tdtype = self.dtype_to_tdtype(dtype) 
        ######################################### 

    def dtype_to_tdtype(self,dtype) :
        if   dtype == "float" :  tdtype = torch.float
        elif dtype == "double" : tdtype = torch.double
        return tdtype

ATTR = Attr()

class Variables :

    def __init__(self) :
        self.data = []

    def register(self,param) :
        self.data.append(param)

    def set_n_path(self,n_path) :
        for i in range(len(self.data)) :
            self.data[i].set_n_path(n_path)

    def mean(self,n_path) :
        for i in range(len(self.data)) :
            self.data[i].mean(n_path)

    def clear(self) :
        self.data.clear()

VARIABLES = Variables()

class NeurPrc_MC :

    def __call__() :
        return NotImplementedError("") 

    @property
    def model_name() : 
        return NotImplementedError("") 

    @property
    def method_name() : 
        return NotImplementedError("") 

    @property
    def column_names() : 
        return NotImplementedError("") 

    @property
    def n_path(self) :
        return NotImplementedError("") 

    @property
    def file_num(self) :
        return NotImplementedError("") 

    @property
    def dtype(self) :
        return "float" 

    @property
    def prod_size(self) :
        return 200

class NeurPrc_Net :

    def __init__(self,mc) :
        self.n_path = mc.n_path

    @property
    def input_node_num(self) :
        return NotImplementedError("") 

    @property
    def hidden_layer_num(self) :
        return NotImplementedError("") 

    @property
    def node_num_per_layer(self) :
        return NotImplementedError("") 

    @property
    def learning_rate(self) :
        return NotImplementedError("") 

    @property
    def act_fn_type(self) :
        return "relu"

    @property
    def fin_act_fn_type(self) :
        return "softplus"

    @property
    def batch_size(self) :
        return 100

    @property
    def eval_batch_size(self) :
        return 100

    @property
    def chunk_size(self) :
        return 10

    @property
    def eval_chunk_size(self) :
        return 1000

    @property
    def num_workers(self) :
        return 5

    @property
    def print_freq(self) :
        return 100

    @property
    def time_limit(self) :
        return 24

    @property
    def test_freq(self) :
        return 5

    @property
    def best_time(self) :
        return ""

    @property
    def continue_mode(self) :
        return False
