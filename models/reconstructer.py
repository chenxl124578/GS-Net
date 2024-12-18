'''
Created in 2022/7/16
by: Xiaolei Chen

this is AutoEncoder model for reconstruction task
copy from (PyTorch复现版 但有问题) https://github.com/square-1111/3D-Point-Cloud-Modeling
which is rewrite from (TF源码) https://github.com/optas/latent_3d_points

'''
from torch import nn
import torch
from .src.chamfer_distance.chamfer_distance import ChamferDistance

class Encoder(nn.Module):

    def __init__(self,*args):
        """
        An Encoder (recognition network), which mapes inputs onto a latent space

        Args:
        *args : dictionary of parameters
        """

        super(Encoder, self).__init__()

        param = args[0]
        n_filters = param['n_filters']
        verbose = param['verbose']
        filter_sizes = param['filter_sizes']
        stride = param['stride']
        padding = param['padding']
        padding_mode = param['padding_mode']
        b_norm = param['b_norm']
        non_linearity = param['non_linearity']
        pool = param['pool']
        pool_sizes = param['pool_sizes']
        dropout_prob = param['dropout_prob']


        self.symmetry = param['symmetry']
        self.closing = param['closing']
        
            
        if verbose:
            print("Building Encoder")
        
        n_layers = len(n_filters)  # 6
        filter_sizes = filter_sizes * n_layers
        strides = stride * n_layers

        self.model = nn.ModuleList()
        # 'n_filters': [3, 64, 128, 128, 256, bneck_size] 'filter_sizes': [1]
        for i in range(n_layers-1):
        
            self.model.append(
                nn.Conv1d(in_channels=n_filters[i], out_channels=n_filters[i+1],
                            kernel_size=filter_sizes[i], stride=strides[i],
                            padding=padding, padding_mode=padding_mode))
            
            if b_norm:
                self.model.append(
                    nn.BatchNorm1d(num_features=n_filters[i+1]))
        
            if non_linearity is not None:
                self.model.append(non_linearity())
                # layer = (layer)
        
            if pool is not None and pool_sizes is not None:
                if pool_sizes[i] is not None:
                    self.model.append(
                        pool(kernel_size=pool_sizes[i]))
        
            if dropout_prob is not None and dropout_prob > 0:
                self.model.append(nn.Dropout(1 - dropout_prob))
        

    def forward(self, X):

        for layer in self.model:
            X = layer(X)

        if self.symmetry is not None:
            X = self.symmetry(X, axis=-1).values
        
        if self.closing is not None:  # None
            X = self.closing(X)

        return X

class DecoderWithFC(nn.Module):
    
    def __init__(self, *args):
        """
        An Decoder (recognition network), which mapes inputs onto a latent space
        """

        super(DecoderWithFC, self).__init__()

        param = args[0]
        verbose = param['verbose']
        layer_sizes = param['layer_sizes'] 
        n_layers = len(layer_sizes)
        dropout_prob = param['dropout_prob']
        b_norm = param['b_norm']
        non_linearity = param['non_linearity']
        b_norm_finish = param['b_norm_finish']
        

        if verbose:
            print("Building Decoder")

        self.model = nn.ModuleList()
        for i in range(n_layers-2):
            
            self.model.append(
                nn.Linear(in_features=layer_sizes[i], 
                            out_features=layer_sizes[i+1]))
            
            if b_norm:   # SampleNet中为 False
                self.model.append(
                    nn.BatchNorm1d(num_features=layer_sizes[i+1]))
            
            if non_linearity is not None:
                self.model.append(non_linearity())
            
            if dropout_prob is not None and dropout_prob > 0:
                self.model.append(nn.Dropout(1 - dropout_prob))

        self.model.append(
            nn.Linear(in_features=layer_sizes[-2], 
                        out_features=layer_sizes[-1]).cuda())

        if b_norm_finish: 
            self.model.append(
                    nn.BatchNorm1d(num_features=layer_sizes[-1]).cuda())
        
    def forward(self, X):
        for layer in self.model:
            X = layer(X)
        return X

class AutoEncoder(nn.Module):
    """Combining the Encoder and Decoder Architecture"""
    def __init__(self, change_init,*args):

        super(AutoEncoder, self).__init__()

        encoder_args = args[0]
        decoder_args = args[1]

        self.encoder = Encoder(encoder_args)
        self.decoder = DecoderWithFC(decoder_args)
        
        if change_init:   # 改初始化
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    m.weight.data = nn.init.uniform_(m.weight.data)  # weight均匀分布（TF源码是uniform_scaling,pytorch没有）
                    if m.bias is not None:
                        m.bias.data = nn.init.constant(m.bias.data, 0.0) # bias全0
                if isinstance(m,nn.Linear):
                    m.weight.data = nn.init.xavier_uniform_(m.weight.data)  # weight 为 xavier 均匀分布
                    if m.bias is not None:
                        m.bias.data = nn.init.constant(m.bias.data, 0.0) # bias全0
                
    
    def forward(self, X):
        latent = self.encoder(X)
        output = self.decoder(latent)
        return torch.reshape(output, (-1, 2048, 3))

    def get_recon_loss(self, out_pc, in_pc):
        # out_pc and in_pc are B x N x 3 matrices

        cost_p1_p2, cost_p2_p1 = ChamferDistance()(in_pc, out_pc)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + cost_p2_p1
        return loss

    def chamferLossBatch(self, X_batch,Y_batch):
        """
        Chamfer loss between two batch of point cloud
        """
        def chamferDistance(X, Y):
            """
            Calcuate Chamfer Loss between two point clouds.

            d_{CH}(S_1, S_2)=\sum_{x\in S_1}\min_{y \in S_2}\|x-y\|^2_2 + \sum_{y\in S_2}\min_{x \in S_1}\|x-y\|^2_2

            Step 1) Calcuate the distance matrix between point cloud X and Y
            Step 2) a) Find the coordinates of the points in point cloud Y 
                    corresponding to nearest to point cloud X
                    b) Find the coordinates of the points in point cloud X 
                    corresponding to nearest to point cloud Y
            Step 3) Calculate norm and sum distances
            """

            dist_mat = torch.cdist(X, Y)
            x_min_idx = dist_mat.min(1).indices
            y_min_idx = dist_mat.min(0).indices

            cham_y = ((Y - X[x_min_idx])**2).sum()
            cham_x = ((X - Y[y_min_idx])**2).sum()

            return (cham_y + cham_x)/X.shape[0]

        loss = 0
        for i in range(X_batch.shape[0]):
            loss += chamferDistance(X_batch[i],Y_batch[i])   # batch中的样本loss求和
        # return loss
        return loss / X_batch.shape[0]


if __name__ == "__main__":
    bneck_size = 128
    encoder_args = {'n_filters': [3, 64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'stride': [1],
                    'b_norm': True,
                    'verbose': True,
                    'non_linearity' : nn.ReLU,
                    'weight_decay' : 0.001,   # ?
                    'dropout_prob' : None,
                    'pool' : nn.AvgPool1d,
                    'pool_sizes' : None,
                    'padding' : 0, 
                    'padding_mode' : 'zeros',
                    'conv_op' : nn.Conv1d,
                    'symmetry' : torch.max, 
                    'closing' : None
                    }
    decoder_args = {'layer_sizes': [128, 256, 256, 2048*3],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True,
                    'non_linearity' : nn.ReLU,
                    'dropout_prob' : None
                    }
    model = AutoEncoder(encoder_args,decoder_args)
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        print(parameters)