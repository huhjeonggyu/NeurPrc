from NeurPrc.all import *

class BS_Exact_MC(NeurPrc_MC) :

    @property
    def model_name(self) : 
        return 'black'

    @property
    def method_name(self) : 
        return 'exact'

    @property
    def column_names(self) : 
        return ['lnK/sqrt(T)','T','vol','prc']

    @property
    def n_path(self) :
        return int(1e6)

    @property
    def file_num(self) :
        return 100

    def __call__(self,n_path) :

        VARIABLES.clear()

        T   = TimeParameter(1.5,300)
        vol = SpaceParameter(0.01,0.5)
        lnK = SpaceParameter(-3.*sqrt(T),3.*sqrt(T))

        VARIABLES.set_n_path(n_path)

        S0 = 1.
        W = BrownianMotion(T)
        S = S0*exp( -0.5*vol**2*T + vol*W )

        c = Product( relu(S-exp(lnK)) )

        VARIABLES.mean(n_path)

        x1 = lnK/sqrt(T)
        x2 = T
        x3 = vol

        y1 = c

        return make_dataset([x1,x2,x3],[y1])

class BS_Euler_MC(NeurPrc_MC) :

    @property
    def model_name(self) : 
        return 'black'

    @property
    def method_name(self) : 
        return 'euler'

    @property
    def column_names(self) : 
        return ['lnK/sqrt(T)','T','vol','prc']

    @property
    def n_path(self) :
        return int(1e5)

    @property
    def file_num(self) :
        return 100

    def __call__(self,n_path) :

        VARIABLES.clear()

        T   = TimeParameter(1.5,300)
        vol = SpaceParameter(0.01,0.5)
        lnK = SpaceParameter(-3.*sqrt(T),3.*sqrt(T))

        S = StochasticProcess(1.)
        t = 0.

        VARIABLES.set_n_path(n_path)

        dt = TimeIncrement(T)
        dW = BrownianMotionIncrement(T)
        for k in range(300) :
            S *= exp( -0.5*vol**2*dt(t) + vol*dW(t) )
            t += dt.dt

        c = Product( relu(S-exp(lnK)) )

        VARIABLES.mean(n_path)

        x1 = lnK/sqrt(T)
        x2 = T
        x3 = vol

        y1 = c

        return make_dataset([x1,x2,x3],[y1])

class BS_PrcNet(NeurPrc_Net) :

    @property
    def input_node_num(self) :
        return 3

    @property
    def hidden_layer_num(self) :
        return 2

    @property
    def node_num_per_layer(self) :
        return 1000

    @property
    def learning_rate(self) :
        return 1e-5

    @property
    def continue_mode(self) :
        return True
