import numpy as np
import torch
from NeurPrc.init import *
from NeurPrc.func import sqrt

class Variable :

    def __init__(self,data) :
        self.data = data
        VARIABLES.register(self)

    def set_n_path(self,n_path) :
        self.data = self.data.repeat(n_path)

    def mean(self,n_path) :
        self.data = self.data.reshape(n_path,ATTR.prod_size)
        self.data = torch.mean(self.data,axis=0)

    @property
    def shape(self) :
        return self.data.shape

    def __len__(self) :
        return len(self.data)

    def __add__(self,other) :
        if np.isscalar(other) : return self.data + other
        else                  : return self.data + other.data

    def __radd__(self,other) :
        return self + other

    def __mul__(self,other) :
        if np.isscalar(other) : return self.data * other
        else                  : return self.data * other.data

    def __rmul__(self,other) :
        return self * other

    def __pow__(self,other) :
        if np.isscalar(other) : return self.data ** other
        else                  : return self.data ** other.data

class SpaceParameter(Variable) :

    def __init__(self,lower,upper) :

        data = (upper-lower) * torch.rand(ATTR.prod_size,device=ATTR.device,dtype=ATTR.tdtype) + lower
        Variable.__init__(self,data)

class TimeParameter(Variable) :

    def __init__(self,upper,m) :

        self.dt = upper/m
        data = self.dt * (1+torch.randint(m,size=(ATTR.prod_size,),device=ATTR.device,dtype=ATTR.tdtype))
        Variable.__init__(self,data)

class StochasticProcess(Variable) : 

    def __init__(self,c) :
        data = torch.full([ATTR.prod_size],c,device=ATTR.device,dtype=ATTR.tdtype)
        Variable.__init__(self,data)

class TimeIncrement :

    def __init__(self,T) :
        self.dt = T.dt
        self.T = T.data

    def __call__(self,t) :
        tmp = torch.full_like(self.T,self.dt)
        tmp[t>=self.T] = 0.
        return tmp

class BrownianMotion : 

    def __init__(self,T) :
        self.data = torch.randn(len(T),device=ATTR.device,dtype=ATTR.tdtype)*sqrt(T)

class BrownianMotionIncrement :

    def __init__(self,T) :
        self.dt = T.dt
        self.length = len(T)
        self.T = T.data

    def __call__(self,t) :
        tmp = torch.randn(self.length,device=ATTR.device,dtype=ATTR.tdtype)*sqrt(self.dt)
        tmp[t>=self.T] = 0.
        return tmp

class Product(Variable) :

    def __init__(self,data) :
        Variable.__init__(self,data)
