from pathlib import Path

import torch
import torch.nn as nn

class LN(nn.module):
    def __init__(self, number_features, eps=1e-8):
        super(LN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones([1, number_features, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, number_features, 1, 1]))

    def forward(self, x):
        """
        前向过程
        output = (x- μ) / σ * γ + β
        :param x: [N, C, H, W]
        :return: [N, C, H, W]
        """
        _mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)  # [N, 1, 1, 1]
        _var = torch.var(x, dim=(1, 2, 3), keepdim=True)    # [N, 1, 1, 1]
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gamma + self.beta

if __name__ == '__main__':
    torch.manual_seed(1)
    path_dir = Path("./output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    bn = LN(num_features=12)
    bn.to(device)  # 只针对子模型或者参数进行转换

    # 模拟训练过程
    bn.train()
    xs = [torch.randn(8, 12, 32, 32).to(device) for _ in range(10)]
    for _x in xs:
        bn(_x)

    print(bn.running_mean.view(-1))
    print(bn.running_var.view(-1))

    # 模拟推理过程
    bn.eval()
    _r = bn(xs[0])
    print(_r.shape)

    bn = bn.cpu()
    # 模拟模型保存
    torch.save(bn, str(path_dir / "ln_model.pkl"))
    # state_dict: 获取当前模块的所有参数(parameter + register_buffer)
    torch.save(bn.state_dict(), str(path_dir / "ln_params.pkl"))
    # pt结构的保存
    traced_script_module = torch.jit.trace(bn.eval().cpu(), xs[0].cpu())
    traced_script_module.save('./output/models/ln_model.pt')

    # 模拟模型恢复
    bn_model = torch.load(str(path_dir / "ln_model.pkl"), map_location='cpu')
    bn_params = torch.load(str(path_dir / "ln_params.pkl"), map_location='cpu')
    print(bn_params)