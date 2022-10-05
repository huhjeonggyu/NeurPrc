import traceback
from NeurPrc.init import ATTR
from NeurPrc.generate import generate
from NeurPrc.train import train

def train_while_generating(mc,net,rank,size) :
    try :
        if size >= 2 :
            if rank < size-1 : 
                ATTR.init(mc,True,rank,size)
                generate(mc,rank,size-1) 
            else : 
                ATTR.init(mc,False,rank,size)
                train(net)
    except Exception as e :
        print(traceback.format_exc())
        with open("finish","wt") as f : 
            f.write("1")

def train_only(mc,net,rank,size) :
    if rank == 0 :
        ATTR.init(mc,False,rank,size)
        train(net)

def generate_only(mc,rank,size) :
    try :
        ATTR.init(mc,True,rank,size)
        generate(mc,rank,size) 
    except Exception as e :
        print(traceback.format_exc())
        with open("finish","wt") as f : 
            f.write("1")

def generate_test(mc,n_path,rank,size) :
    try :
        ATTR.init(mc,False,rank,size)
        return mc(n_path)
    except Exception as e :
        print(traceback.format_exc())
        with open("finish","wt") as f : 
            f.write("1")
