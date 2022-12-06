import os
import sys
import argparse
from statistics import mean,stdev
import torch
import torch.nn.functional as F
import scipy
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import scipy.sparse as sp
import numpy as np
import torch_geometric
from torch_geometric.utils import get_laplacian
sys.path.append(os.path.join(os.path.dirname(__file__), "diffusion_net/")) 
from load_data import get_dataset, split_data
from layers import TIDE_net
from funcs import get_optimizer, get_laplacian_selfloop, sparse_mx_to_torch_sparse_tensor


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Pubmed')  
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=64) 
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--k', type=int, default=64)   
    parser.add_argument('--iteration', type=int, default=1) 
    parser.add_argument('--num_blocks', type=int, default=1) 
    parser.add_argument('--single_t', type=int, default=0)
    parser.add_argument('--Lap_feat', type=int, default=1)
    parser.add_argument('--show_plot', type=int, default=0)
    parser.add_argument('--MLP', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adamax')
    parser.add_argument('--lap_type', type=str, default='without_sl') # 'sym', 'rw' 'with_sl'  'without_sl'
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--weight_decay', type=float, default=0)    
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    
    
    args = parser.parse_args()
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    
    ##  load dataset
    ds_name=args.dataset
       
    
    if ds_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS',]:
        dataset=split_data(get_dataset(ds_name))
        data=dataset[0]

            

    num_nodes=data.num_nodes
    C_in=data.x.shape[-1]
    n_class=dataset.num_classes
    
    
    
    ## compute laplacian
    if args.lap_type in ['with_sl','without_sl']:
        L = get_laplacian_selfloop(data,args.lap_type)
    else:
        L = get_laplacian(data.edge_index, data.edge_weight, num_nodes=num_nodes, normalization=args.lap_type)
            
    
    L_sparse = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
    
    
    evals, evecs = scipy.sparse.linalg.eigs(L_sparse, k=args.k , M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None)
    
    evals=torch.tensor(evals.real)
    evecs=torch.tensor(evecs.real)
    
    
    ## convert L to sparse tensor
    L = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(L_sparse)).to(device)
    
  
    ## Compute Lap feature
            
    gradX=torch.zeros(10, 10).to(device)
    
    if args.Lap_feat:
        idx = torch.LongTensor([1,0])
        # idx = torch.LongTensor([1,0]).to(device)
        v = torch.cat((torch.ones(data.edge_index.shape[1], device=device), - torch.ones(data.edge_index.shape[1], device=device)))
        i = torch.transpose(torch.cat((data.edge_index, data.edge_index.index_select(0, idx)), dim = 1).to(device), 0,1)
        # s = torch.sparse_coo_tensor(i, v)
        
        gradX = torch.zeros((num_nodes, num_nodes)).to(device)
        
        for ind in i:
            gradX[ind[0], ind[1]] = 1
            
    gradY = gradX
    

    
    def train(epoch, optimizer, model, data, mass, L, evals, evecs, gradX, gradY):
                    
        model.train()
        optimizer.zero_grad()
        
        # Apply the model
        out = model(epoch, data.x, data.edge_index, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
    
        # Evaluate loss
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])   

        loss.backward()    # Back Propagation
        optimizer.step()   # Gardient Descent

        
        return loss
    
    
    
    @torch.no_grad()
    def test(epoch, model, data, mass, L, evals, evecs, gradX, gradY):
        model.eval()
    
    
        with torch.no_grad():    
            logits, accs = model(epoch, data.x, data.edge_index, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY), []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    
        return accs
    
    
    
    ## Run mode
    
    train_ls = []
    val_ls = []
    test_ls = []
                    
    for _ in range(args.iteration):

    
    
    
    
        model = TIDE_net(k=args.k, C_in=C_in,C_out=n_class, C_width=args.hidden_channels, num_nodes = num_nodes ,N_block=args.num_blocks, single_t=args.single_t, use_gdc=args.use_gdc,
                            last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                            diffusion_method='spectral',
                            with_Lap_feat = args.Lap_feat, 
                            with_MLP = args.MLP,
                            grad_matrix = gradX,
                            dropout=True,
                            device = device)
    
        model = model.to(device)
    
    
        parameters = [p for p in model.parameters() if p.requires_grad]
    
        parameters_name= [ name for (name, param) in model.named_parameters() if param.requires_grad]
    
        # Move to device
        data = data.to(device)
        x = data.x.to(device)
        mass=torch.ones(num_nodes).to(device)
        evals=evals.to(device)
        evecs=evecs.to(device)
        gradX=gradX.to(device)
        gradY=gradY.to(device)
    
    
        optimizer = get_optimizer(args.optimizer, parameters, lr = args.lr, weight_decay=args.weight_decay)      
    
    
        total_train=[]
        total_test=[]
        total_val=[]
    
        best_epoch = train_acc = val_acc = test_acc = 0

        for epoch in range(1, args.epochs + 1):
           
            loss = train(epoch, optimizer, model, data, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
            
            tmp_train_acc, tmp_val_acc, tmp_test_acc = test(epoch, model, data, mass=mass, L=L, evals=evals, evecs=evecs,  gradX=gradX, gradY=gradY)
            if tmp_val_acc > val_acc:
                best_epoch = epoch
                train_acc = tmp_train_acc
                val_acc = tmp_val_acc
                test_acc = tmp_test_acc
                  
            
                total_train.append(train_acc)
                total_val.append(val_acc)
                total_test.append(test_acc)
                
                # log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
                print(f'Step {epoch}: ' f' Loss: {float(loss):.4f}, Train Acc: {train_acc:.4f},'f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
                
    
        train_ls.append(train_acc)
        val_ls.append(val_acc)
        test_ls.append(test_acc)
    
    
    print(f'Average: Train Acc: {mean(train_ls)*100:2.2f}, Val Acc: {mean(val_ls)*100:2.2f}, Test Acc: {mean(test_ls)*100:2.2f}')
    
    if args.iteration>1:
        print(f'Test Acc: {mean(test_ls)*100:2.2f}\xB1{stdev(test_ls):.1e}')

        
    ##Plot
    
    if args.show_plot:
        import matplotlib.pyplot as plt
        plt.close()
        
        plt.plot(total_train,'-')
        plt.plot(total_val,'-')
        plt.plot(total_test,'-')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Valid','Test'])
        plt.title('Accuracy')
        
        plt.show()

if __name__ == "__main__":
    main()












