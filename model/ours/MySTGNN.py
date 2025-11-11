import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model.ours.Architecture import *


class MySTGNN(nn.Module):
    """
    return: H: (batch_size, node_num, lag) or H2: (batch_size, node_num, pred_len)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.Architecture == "Parallel":
            self.STGNN = STGNN_Parallel(args)
        elif args.Architecture == "Series_ST":
            self.STGNN = STGNN_Series_ST(args)
        elif args.Architecture == "Series_TS":
            self.STGNN = STGNN_Series_TS(args)
        elif args.Architecture == "Series_STS":
            self.STGNN = STGNN_Series_STS(args)
        else:
            raise Exception("No such architecture! must be 'Parallel' or 'Series_ST' or 'Series_TS' or 'Series_STS'!")

    def reconstruct(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        H = self.STGNN(X, A)
        'H: (batch_size, node_num, lag)'

        return H

    def forecast(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        H = self.STGNN(X, A)
        'H: (batch_size, node_num, lag)'
        # H2 = H[:, :, : self.args.pred_len]
        H2 = H[:, :, -self.args.pred_len:]
        """其实应该是 H2 = H[:, :, -self.args.pred_len:], 怪不得TCN效果不好"""
        'H2: (batch_size, node_num, pred_len)'

        return H2

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)

        :return: H: (batch_size, node_num, lag) or H2: (batch_size, node_num, pred_len)
        """
        if self.args.BaseOn == "reconstruct":
            return self.reconstruct(X, A)
            'H: (batch_size, node_num, lag)'
        elif self.args.BaseOn == "forecast":
            return self.forecast(X, A)
            'H2: (batch_size, node_num, pred_len)'
        else:
            raise Exception("No such BaseOn! must be 'reconstruct' or 'forecast'!")



