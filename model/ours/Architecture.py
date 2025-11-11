import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from model.ours.spatial_block import *
from model.ours.temporal_block import *


class MLP_Concat(nn.Module):
    def __init__(self, args):
        super(MLP_Concat, self).__init__()
        self.args = args
        self.W = Parameter(torch.Tensor(args.fusion_hidden_dim, args.node_num * 2))
        self.b = Parameter(torch.Tensor(args.fusion_hidden_dim, 1))
        self.E = Parameter(torch.Tensor(args.node_num, args.fusion_hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.normal_(self.b)
        torch.nn.init.xavier_uniform_(self.E)

    def forward(self, H_S, H_T):
        """
        :param H_S: (batch_size, node_num, lag)
        :param H_T: (batch_size, node_num, lag)
        """
        if H_S.shape != H_T.shape:
            raise Exception("H_S and H_T must have the same shape! check the pred_len > lag? STGNN_Parallel is not suitable for this situation!")
        H = torch.cat((H_S, H_T), dim=1)
        'H: (batch_size, node_num * 2, lag)'
        H = torch.matmul(self.W, H) + self.b
        'H: (batch_size, fusion_hidden_dim, lag)'
        H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
        'H: (batch_size, fusion_hidden_dim, lag)'
        H = nn.Dropout(p=self.args.dropout)(H)
        'H: (batch_size, fusion_hidden_dim, lag)'
        H = torch.matmul(self.E, H)
        'H: (batch_size, node_num, lag)'

        return H


class Gate_Weight(nn.Module):
    def __init__(self, args):
        super(Gate_Weight, self).__init__()
        self.args = args
        self.W1 = Parameter(torch.Tensor(args.node_num, args.node_num))
        self.W2 = Parameter(torch.Tensor(args.node_num, args.node_num))
        self.b = Parameter(torch.Tensor(args.node_num, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)
        torch.nn.init.normal_(self.b)

    def forward(self, H_S, H_T):
        """
        :param H_S: (batch_size, node_num, lag)
        :param H_T: (batch_size, node_num, lag)
        """
        if H_S.shape != H_T.shape:
            raise Exception("H_S and H_T must have the same shape! check the pred_len > lag? STGNN_Parallel is not suitable for this situation!")
        Z = torch.matmul(self.W1, H_S) + torch.matmul(self.W2, H_T) + self.b
        'Z: (batch_size, node_num, lag)'
        Z = F.sigmoid(Z)
        'Z: (batch_size, node_num, lag)'
        H = Z * H_S + (1 - Z) * H_T
        'H: (batch_size, node_num, lag)'

        return H


class Add_Minus(nn.Module):
    def __init__(self, args):
        super(Add_Minus, self).__init__()
        self.args = args
        self.W = Parameter(torch.Tensor(args.node_num, args.node_num * 2))
        self.b = Parameter(torch.Tensor(args.node_num, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.normal_(self.b)

    def forward(self, H_S, H_T):
        """
        :param H_S: (batch_size, node_num, lag)
        :param H_T: (batch_size, node_num, lag)
        """
        if H_S.shape != H_T.shape:
            raise Exception("H_S and H_T must have the same shape! check the pred_len > lag? STGNN_Parallel is not suitable for this situation!")
        H1 = H_S + H_T
        'H1: (batch_size, node_num, lag)'
        H2 = H_S - H_T
        'H2: (batch_size, node_num, lag)'
        H = torch.cat((H1, H2), dim=1)
        'H: (batch_size, node_num * 2, lag)'
        H = torch.matmul(self.W, H) + self.b
        'H: (batch_size, node_num, lag)'
        H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
        'H: (batch_size, node_num, lag)'
        H = nn.Dropout(p=self.args.dropout)(H)
        'H: (batch_size, node_num, lag)'

        return H


class spatail_block(nn.Module):
    def __init__(self, args):
        """
        MAdjGCN or MAdjGCN_Lite or CMTS_GCN or GCN_s or Muti_S_GAT or None        【【【MAdjGCN is G3CN, CMTS_GCN is multi-layer G3CN】】】, 
        input X(batch_size, node_num, lag) and A(node_num, node_num), output H(batch_size, node_num, lag)

        Args:
            args:
        """
        super(spatail_block, self).__init__()
        self.args = args
        if args.spatial_method == 'MAdjGCN':
            self.spatial_block = MAdjGCN(args)
        elif args.spatial_method == 'MAdjGCN_Lite':
            self.spatial_block = MAdjGCN_Lite(args)
        elif args.spatial_method == 'CMTS_GCN':
            self.spatial_block = CMTS_GCN(args)
        elif args.spatial_method == 'GCN_s':
            self.spatial_block = GCN_s(args)
        elif args.spatial_method == 'Muti_S_GAT':
            self.spatial_block = Muti_S_GAT(args)
        elif args.spatial_method == 'GIN':
            self.spatial_block = GIN(args)
        elif args.spatial_method == 'SGC':
            self.spatial_block = SGC(args)
        elif args.spatial_method == 'GPRGNN':
            self.spatial_block = GPRGNN(args.GPRGNN_K, args.self_edge)
        elif args.spatial_method == 'H2GCN':
            if args.Architecture == "Series_TS" and args.temporal_method == "Informer":
                input_lag = args.pred_len
            else:
                input_lag = args.lag
            self.spatial_block = H2GCN(lag=input_lag, feature_dim=args.H2GCN_embed_dim, output_lag=input_lag,
                                       H2GCN_round_K=args.H2GCN_round_K, dropout=args.dropout, if_self_edge=args.self_edge)
            if args.graph_if_norm_A: raise Exception("H2GCN does not support graph_if_norm_A!")
        elif args.spatial_method == 'None':
            self.spatial_block = Nothing_to_do_S()
        else:
            raise Exception("No such spatial method!")

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)

        :return: H: (batch_size, node_num, lag)
        """
        H = self.spatial_block(X, A)
        'H: (batch_size, node_num, lag)'

        return H


class temporal_block(nn.Module):
    def __init__(self, args):
        """
        GRU or TCN or Muti_T_GAT or None，输入X(batch_size, node_num, lag)，输出H(batch_size, node_num, lag or pred_len or recurrent_num*pred_step)
        if pred_len > lag, forcast by recurrently using GRU or TCN, each time predict pred_step time steps

        Args:
            args:
        """
        super(temporal_block, self).__init__()
        self.args = args
        if args.temporal_method == 'GRU':
            self.temporal_block = GRU(input_size=args.node_num, hidden_size=args.GRU_hidden_num,
                                      num_layers=args.GRU_layers, dropout=args.dropout)
            if args.GRU_hidden_num != args.node_num:
                self.W = Parameter(torch.Tensor(args.node_num, args.GRU_hidden_num))
                self.b = Parameter(torch.Tensor(args.node_num, 1))
                self.reset_parameters()
        elif args.temporal_method == 'TCN':
            self.temporal_block = TCN(num_inputs=args.node_num, num_channels=args.TCN_layers_channels,
                                      kernel_size=args.TCN_kernel_size, dropout=args.dropout)
            if args.TCN_layers_channels[-1] != args.node_num:
                self.W = Parameter(torch.Tensor(args.node_num, args.TCN_layers_channels[-1]))
                self.b = Parameter(torch.Tensor(args.node_num, 1))
                self.reset_parameters()
        elif args.temporal_method == 'Muti_T_GAT':
            self.temporal_block = Muti_T_GAT(args)
        elif args.temporal_method == 'None':
            self.temporal_block = Nothing_to_do_T()
        else:
            raise Exception("No such temporal method!")

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.normal_(self.b)

    def forward(self, X):
        """
        :param X: (batch_size, node_num, lag)

        :return: H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)
        """
        if self.args.BaseOn == 'forecast' and self.args.pred_len > self.args.lag and self.args.temporal_method != 'None':
            recurrent_num = math.ceil(self.args.pred_len / self.args.pred_step)
            H_step_in = X
            'H_step: (batch_size, node_num, lag)'
            for i in range(recurrent_num):
                H_step_out = self.temporal_block(H_step_in)
                'H_step: (batch_size, hidden_size/TCN_layers_channels[-1], lag)'
                if H_step_out.shape[1] != X.shape[1]:
                    H_step_out = torch.matmul(self.W, H_step_out) + self.b
                'H_step: (batch_size, node_num, lag)'
                H_step_out = F.leaky_relu(H_step_out, negative_slope=self.args.LeakyReLU_slope)
                'H_step: (batch_size, node_num, lag)'
                H_step_out = H_step_out[:, :, -self.args.pred_step:]
                'H_step: (batch_size, node_num, pred_step)'
                H_step_in = torch.cat((H_step_in[:, :, self.args.pred_step:], H_step_out), dim=2)
                'H_step: (batch_size, node_num, lag)'
                if i == 0:
                    H = H_step_out
                else:
                    H = torch.cat((H, H_step_out), dim=2)
                    'H: (batch_size, node_num, pred_len) or (batch_size, node_num, recurrent_num*pred_step)'
        else:
            H = self.temporal_block(X)
            'H: (batch_size, hidden_size/TCN_layers_channels[-1], lag)'
            if H.shape[1] != X.shape[1]:
                H = torch.matmul(self.W, H) + self.b
            'H: (batch_size, node_num, lag)'
            H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
            'H: (batch_size, node_num, lag)'

        return H


class STGNN_Parallel(nn.Module):
    def __init__(self, args):
        super(STGNN_Parallel, self).__init__()
        self.args = args

        self.spatial_block = spatail_block(args)
        self.temporal_block = temporal_block(args)

        if args.fusion_method == 'MLP_Concat':
            self.fusion_block = MLP_Concat(args)
        elif args.fusion_method == 'Gate_Weight':
            self.fusion_block = Gate_Weight(args)
        elif args.fusion_method == 'Add_Minus':
            self.fusion_block = Add_Minus(args)
        else:
            self.fusion_block = None
            print("No such fusion method!")

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        # spatial
        H_S = self.spatial_block(X, A)
        'H_S: (batch_size, node_num, lag)'
        # temporal
        H_T = self.temporal_block(X)
        'H_T: (batch_size, node_num, lag)'
        if H_S.shape != H_T.shape:
            raise Exception("H_S and H_T must have the same shape! Check the pred_len > lag? STGNN_Parallel is not suitable for this situation!")
        # fusion
        if self.fusion_block is not None:
            H = self.fusion_block(H_S, H_T)
            'H: (batch_size, node_num, lag)'
        else:
            H = (H_S + H_T) / 2
            'H: (batch_size, node_num, lag)'

        return H


class STGNN_Series_ST(nn.Module):
    def __init__(self, args):
        super(STGNN_Series_ST, self).__init__()
        self.args = args

        self.spatial_block = spatail_block(args)
        self.temporal_block = temporal_block(args)

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        # spatial
        if self.args.block_residual == 0:
            H = self.spatial_block(X, A)
            'H: (batch_size, node_num, lag)'
        elif self.args.block_residual != 0:
            H = self.spatial_block(X, A) + self.args.block_residual * X
            'H: (batch_size, node_num, lag)'
        # temporal
        H = self.temporal_block(H)
        'H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)'

        return H


class STGNN_Series_TS(nn.Module):
    def __init__(self, args):
        super(STGNN_Series_TS, self).__init__()
        self.args = args

        self.spatial_block = spatail_block(args)
        self.temporal_block = temporal_block(args)

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        # temporal
        H = self.temporal_block(X)
        'H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)'
        # if self.args.block_residual != 0:
        #     H = H + self.args.block_residual * X
        # spatial
        if self.args.block_residual == 0:
            H = self.spatial_block(H, A)
            'H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)'
        elif self.args.block_residual != 0:
            H = self.spatial_block(H, A) + self.args.block_residual * H
            'H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)'

        return H


class STGNN_Series_STS(nn.Module):
    def __init__(self, args):
        super(STGNN_Series_STS, self).__init__()
        self.args = args

        self.spatial_block = spatail_block(args)
        self.temporal_block = temporal_block(args)

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        # spatial
        if self.args.block_residual == 0:
            H = self.spatial_block(X, A)
            'H: (batch_size, node_num, lag)'
        elif self.args.block_residual != 0:
            H = self.spatial_block(X, A) + self.args.block_residual * X
            'H: (batch_size, node_num, lag)'
        # temporal
        H = self.temporal_block(H)
        'H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)'
        # spatial
        if self.args.block_residual == 0:
            H = self.spatial_block(H, A)
            'H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)'
        elif self.args.block_residual != 0:
            H = self.spatial_block(H, A) + self.args.block_residual * H
            'H: (batch_size, node_num, lag or pred_len or recurrent_num*pred_step)'

        return H















