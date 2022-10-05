import torch.nn as nn

class Model(nn.Module) :
    
    def __init__(self,node_num_per_layer,hidden_layer_num,act_fn_type,fin_act_fn_type) :

        super().__init__()

        self.layer_i  = nn.Linear(3,node_num_per_layer)

        self.layers_h = nn.ModuleList()
        for _ in range(hidden_layer_num) :
            self.layers_h.append( nn.Linear(node_num_per_layer,node_num_per_layer) )

        self.layer_o  = nn.Linear(node_num_per_layer,1)

        if   act_fn_type == "relu" :
            self.act_fn = nn.ReLU()
        elif act_fn_type == "leaky_relu" :
            self.act_fn = nn.LeakyReLU()
        elif act_fn_type == "ELU" :
            self.act_fn = nn.ELU()

        if   fin_act_fn_type == "softplus" :
            self.fin_act_fn = nn.Softplus()
        elif fin_act_fn_type == "identity" :
            self.fin_act_fn = nn.Identity()

    def forward(self,x) :

        x = self.layer_i(x)
        x = self.act_fn(x)

        for layer_h in self.layers_h :
            x = layer_h(x)
            x = self.act_fn(x)

        x = self.layer_o(x)
        x = self.fin_act_fn(x)
        return x
