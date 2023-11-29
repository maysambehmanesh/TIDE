import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np   
    
from early_stop_solver import EarlyStopInt
from block_constant import ConstantODEblock
from function_laplacian_diffusion import LaplacianODEFunc
from torch_geometric.nn import GCNConv





class GCN_diff(torch.nn.Module):
    def __init__(self, use_gdc, in_channels,hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels, cached=True, normalize=not use_gdc)
    
    def forward(self,x,edge_index,edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()

        return x
    

class Time_derivative_diffusion(nn.Module):
    def __init__(self, n_block, GCN_opt, k, C_inout, num_nodes, single_t, method='spectral'):
        super(Time_derivative_diffusion, self).__init__()
        self.C_inout = C_inout
        self.k = k
        self.single_t = single_t
        
        # same t for all channels
        if self.single_t:
            self.diffusion_time = nn.Parameter(torch.Tensor(1))
        else:
        # learnable t
            self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  

       
        self.Conv_layer = GCN_diff(GCN_opt, C_inout, C_inout)

        
        self.method = method # one of ['spectral', 'implicit_dense']
        self.num_nodes = num_nodes 
        
        nn.init.constant_(self.diffusion_time, 0.0)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
    
    def reset_parameters(self):
        self.Conv_layer.reset_parameters()            
        nn.init.constant_(self.diffusion_time, 0.0)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.betta = nn.Parameter(torch.tensor(0.0))
        
        
        
    def forward(self, x, edge_index, L, mass, evals, evecs):
         

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':
              
            # Transform to spectral
            x_spec = torch.matmul(torch.transpose(evecs,1,0),x)
            
            # Diffuse
            time = self.diffusion_time
            
            # Same t for all channels
            if self.single_t:
                dim = x.shape[1]
                time = time.repeat(dim)

            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            
            x_diffuse_spec = diffusion_coefs * x_spec

            x_diffuse_spec = x_diffuse_spec -(self.alpha)*x_spec
            x_diffuse = torch.matmul(evecs, x_diffuse_spec)
            x_diffuse = x_diffuse + (self.betta)*x

            # x_diffuse = self.Conv_layer(x_diffuse, edge_index, edge_weight=None).relu()    # with A
            
            
            x_diffuse = self.Conv_layer(x_diffuse, L._indices(), edge_weight=L._values()).relu()   # with L

                      
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            L=torch.tensor(L.todense())
            mat_dense = L.unsqueeze(1).expand(self.C_inout, V, V).clone()
            
            # mat_dense = L.to_dense().expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse

            

class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )



    
class TIDE_block(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, n_block, k, C_in, C_width, C_out, mlp_hidden_dims, num_nodes,
                 dropout=True,
                 diffusion_method='spectral',
                 single_t = False,
                 use_gdc = [],
                 with_MLP=True,
                 device='cpu'):
        super(TIDE_block, self).__init__()

        # Specified dimensions
        self.k = k
        self.C_width = C_width
        self.C_in = C_in
        self.C_out = C_out
        self.mlp_hidden_dims = mlp_hidden_dims
        self.single_t = single_t
        self.use_gdc = use_gdc
        self.dropout = dropout
        self.with_MLP = with_MLP
        self.num_nodes = num_nodes
        self.n_block = n_block
        self.device = device
        self.MLP_C = 2*self.C_width

        self.diff_derivative = Time_derivative_diffusion(self.n_block, self.use_gdc, self.k, self.C_width, self.num_nodes, self.single_t, method=diffusion_method)      
        

        # With MLP
        if self.with_MLP:
            self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)
      
    def forward(self, epoch, x_in, x_original, edge_index, mass, L, evals, evecs, x0):

        # Manage dimensions
        if x_in.shape[-1] != self.C_width:
            raise ValueError("Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(x_in.shape, self.C_width))

        x_diff_derivative = self.diff_derivative(x_in, edge_index, L, mass, evals, evecs)   

        x_diffuse= x_diff_derivative
            

        if self.with_MLP:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        if self.with_MLP:
            x0_out = self.mlp(feature_combined)
            x0_out = x0_out + x_in 
        else:
            x0_out = x_diffuse + x_in 

        return x0_out      


class TIDE_net(nn.Module):

    def __init__(self,k, C_in, C_out, C_width=128, N_block=4, single_t=0, use_gdc=[], num_nodes=[], 
                 last_activation=None, mlp_hidden_dims=None, dropout=True, with_MLP=True, 
                 diffusion_method='spectral', device='cpu'):   
        super(TIDE_net, self).__init__()

        # Basic parameters
        self.k = k
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.num_nodes = num_nodes
        self.single_t = single_t
        self.use_gdc = use_gdc
        self.device = device
   

        # Outputs
        self.last_activation = last_activation
        
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        
        #MLP
        self.with_MLP = with_MLP
        
       
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
            
        # TIDE blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = TIDE_block(n_block = i_block+1,
                                            k = k,
                                            C_in= C_in,
                                            C_width = C_width, 
                                            C_out=C_out,
                                            mlp_hidden_dims = mlp_hidden_dims,
                                            num_nodes = num_nodes,
                                            dropout = dropout,
                                            diffusion_method = diffusion_method,
                                            single_t = single_t,
                                            use_gdc = use_gdc,
                                            with_MLP = with_MLP,
                                            device = self.device)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, epoch, x_in, edge_index, mass, L=None, evals=None, evecs=None, edges=None, faces=None):
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        
        # Apply each of the blocks
        for b in self.blocks:
            # x = self.med_linear(x)
            x = b(epoch, x, x_in, edge_index, mass, L, evals, evecs, x_in)

        # Apply the last linear layer        
        x_out = self.last_lin(x)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        return x_out
