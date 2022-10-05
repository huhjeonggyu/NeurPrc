import torch
import glob
import tables
from NeurPrc.init import ATTR

h5  = lambda x : x.root.df.table
h5i = lambda x,n1,n2 : h5(x).read(n1,n2,field="index")
h5v = lambda x,n1,n2 : h5(x).read(n1,n2,field="values_block_0")

class Dataset(torch.utils.data.Dataset) :

    def __init__(self,save_dir,mod,file_num=None) :

        tdtype = ATTR.tdtype
        prod_size = ATTR.prod_size

        save_dir = save_dir.split("_")
        save_dir = '_'.join(save_dir[:-1])

        self.filelist = glob.glob("data/"+save_dir+"*/*.hdf") 
        if file_num != None :
            self.filelist.sort()
            self.filelist = self.filelist[:file_num]
        self.file_num = len(self.filelist)

        self.mod = mod
        self.tdtype = tdtype 
        self.num_per_file = prod_size*500
        self.len = (self.num_per_file//self.mod)*len(self.filelist)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        fidx = idx//(self.num_per_file//self.mod)
        didx = idx%(self.num_per_file//self.mod)

        file_ = self.filelist[fidx]
        with tables.open_file(file_,mode="r") as h5file :
            tmp = h5v(h5file,self.mod*didx,self.mod*(didx+1))
            tmp = torch.as_tensor(tmp,dtype=self.tdtype)
            x_batch,y_batch = tmp[:,:-1],tmp[:,-1:]
            return (x_batch,y_batch)
