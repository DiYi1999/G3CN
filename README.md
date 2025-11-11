### G3CN




<!-- ### Code, Model, Dataset of [Beyond the Homophily Assumption: Mining Complex Correlations in Time Series via Graph Neural Network](https://www.sciencedirect.com/science/article/xxxxxx)


if it is helpful for your research, you can cite the following paper:
```bibtex
@article{xxxxxx,
title = {Beyond the Homophily Assumption: Mining Complex Correlations in Time Series via Graph Neural Network},
journal = {xxxxxx},
volume = {xxxxxx},
pages = {113380},
year = {xxxxxx},
issn = {xxxxxx},
doi = {xxxxxx},
url = {xxxxxx}
}
```


And we We are currently developing a more comprehensive and enhanced version of our dataset. If you are interested, welcome to follow our updates. We hope it will be helpful to you: [https://diyi1999.github.io/XJTU-SPS/](https://diyi1999.github.io/XJTU-SPS/) -->

<!-- 
Core Code of G3CN:
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

``` -->



