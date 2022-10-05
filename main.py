from NeurPrc.core import *
from NeurExt.BS import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__" :

    #mc = BS_Exact_MC()
    mc = BS_Euler_MC()

    #n_path = int(1e6)
    #x_dat,y_dat = generate_test(mc,n_path,rank,size)

    #generate_only(mc,rank,size)

    net = BS_PrcNet(mc)
    #train_only(mc,net,rank,size)

    train_while_generating(mc,net,rank,size)
