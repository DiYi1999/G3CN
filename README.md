#### G3CN Project

### Code and Data of [Beyond the Homophily Assumption: Mining Complex Correlations in Time Series via Graph Neural Network](https://doi.org/10.1016/j.patcog.2026.113388)

if it is helpful for your research, you can cite the following paper:
```bibtex
@article{DI2026113388,
title = {Beyond the Homophily Assumption: Mining Complex Correlations in Time Series via Graph Neural Network},
journal = {Pattern Recognition},
pages = {113388},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.113388},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326003535},
author = {Yi Di and Fujin Wang and Zhi Zhai and Zhibin Zhao and Xuefeng Chen},
keywords = {multivariate time series, graph neural network, complex multi-sensor system, spatial information, nonlinear correlation}
}
```

#### For ease of use, we extract the core code as follows:

### Core Code One, G3CN:

```python
class Multi_Layer_G3CN(nn.Module):
    def __init__(self, args):
        super(Multi_Layer_G3CN, self).__init__()
        self.args = args
        G3CN_Layer_list = []
        layer_num = len(args.G3CN_Layer_K_nums)
        for i in range(layer_num):
            G3CN_Layer_list.append(G3CN(args, args.G3CN_Layer_K_nums[i]))
        self.G3CN_Layer_list = nn.ModuleList(G3CN_Layer_list)
    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        :return: H: (batch_size, node_num, lag)
        """
        H = X
        for G3CN_Layer in self.G3CN_Layer_list:
            H = G3CN_Layer(H, A) + H * self.args.G3CN_Layer_residual
        return H

class G3CN(nn.Module):
    def __init__(self, args, K=None):
        super(G3CN, self).__init__()
        self.args = args
        self.K = args.K
        if K is not None:
            self.K = K
        self.MAdjG3CNlist = nn.ModuleList([OneAdjG3CN(args) for _ in range(self.K)])
        # last layer for StandardScaler()
        self.last_linear_w = Parameter(torch.Tensor(args.node_num))
        self.last_linear_b = Parameter(torch.Tensor(args.node_num))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.normal_(self.last_linear_w, mean=1.0, std=0.1)
        torch.nn.init.normal_(self.last_linear_b)
    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        :return: H: (batch_size, node_num, lag)
        """
        H = [self.MAdjG3CNlist[i](X, A) for i in range(self.K)]
        H = torch.stack(H, dim=0).sum(dim=0)
        H = self.last_linear_w.unsqueeze(0).unsqueeze(2) * H + self.last_linear_b.unsqueeze(0).unsqueeze(2)
        return H

class OneAdjG3CN(nn.Module):
    def __init__(self, args):
        super(OneAdjG3CN, self).__init__()
        self.args = args
        self.W = Parameter(torch.Tensor(args.node_num, args.node_num))
        self.b = Parameter(torch.Tensor(args.node_num, 1))
        self.v = Parameter(torch.Tensor(args.node_num))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.W, nonlinearity='leaky_relu')
        torch.nn.init.normal_(self.b)
        torch.nn.init.normal_(self.v)
    def forward(self, X, A):
        fixed = torch.zeros_like(self.W, requires_grad=False)
        W = A * self.W + (1 - A) * fixed
        H = torch.matmul(W, X) + self.b
        if self.args.residual_alpha != 0:
            H = (1-self.args.residual_alpha) * torch.matmul(W, X) + self.args.residual_alpha * X + self.b
        H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
        v = self.v.unsqueeze(0).unsqueeze(2)
        H = v * H
        return H
```

### Core Code Two, Computing the adjacency matrix via non-linear metrics:

```python
from minepy import MINE
from minepy import pstats, cstats
from copent import copent
from scipy.stats import kendalltau
from sklearn.feature_selection import mutual_info_regression

def A_w_calculate(args, data_normal):
    if isinstance(data_normal, np.ndarray):
        data = data_normal
    elif isinstance(data_normal, pd.DataFrame):
        data = data_normal.values
    elif isinstance(data_normal, torch.Tensor):
        data = data_normal.cpu().numpy()
    else:
        raise ValueError("data_normal should be np.ndarray or pd.DataFrame")
    ca_len, node_num = data.shape
    A_w = np.zeros((node_num, node_num)).astype(np.float32)
    if args.graph_ca_meth == "MIC":
        # for i in range(0, node_num - 1):
        #     for j in range(i + 1, node_num):
        #         mine = MINE(alpha=args.MIC_alpha, c=args.MIC_c, est="mic_approx")
        #         mine.compute_score(data[:, i], data[:, j])
        #         A_w[i, j] = mine.mic()
        #         A_w[j, i] = A_w[i, j]
        # np.fill_diagonal(A_w, 1)
        data_T = data.T
        mic_p, tic_p =  pstats(data_T, alpha=args.MIC_alpha, c=args.MIC_c, est="mic_e")
        A_w = np.zeros((node_num, node_num)).astype(np.float32)
        triu_idx = np.triu_indices(node_num, k=1)
        A_w[triu_idx] = mic_p
        A_w += A_w.T
        np.fill_diagonal(A_w, 1)
    elif args.graph_ca_meth == "Copent":
        for i in range(0, node_num - 1):
            for j in range(i + 1, node_num):
                data1 = data[:, [i, j]]
                A_w[i, j] = copent(data1)
                A_w[j, i] = A_w[i, j]
        A_w = (A_w - A_w.min()) / (A_w.max() - A_w.min() + 1e-5)
        np.fill_diagonal(A_w, 1)
    elif args.graph_ca_meth == "Kendall":
        for i in range(0, node_num - 1):
            for j in range(i + 1, node_num):
                tau, _ = kendalltau(data[:, i], data[:, j])
                if np.isnan(tau):
                    tau = 0
                A_w[i, j] = tau
                A_w[j, i] = A_w[i, j]
        np.fill_diagonal(A_w, 1)
        A_w = np.abs(A_w)
    elif args.graph_ca_meth == "MutualInfo":
        for i in range(0, node_num - 1):
            X = data[:, i+1:]
            y = data[:, i]
            mi = mutual_info_regression(X, y)
            A_w[i, i+1:] = mi
            A_w[i+1:, i] = mi
        A_w[A_w < 0] = 0
        A_w = A_w / A_w.max() if A_w.max() != 0 else A_w
        np.fill_diagonal(A_w, 1)
    else:
        raise ValueError("method should be MIC or Copent or ...")

    return A_w
```

### Core Code Three, Multi-scale decomposition:

Since this part of the code is relatively lengthy, you can refer directly to [utils/decompose.py](./utils/decompose.py). However, it is not strictly required and can be regarded as an auxiliary component. In ordinary scenarios, the two core code snippets above are sufficient to solve the problem.

