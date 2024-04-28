import torch
import torch.nn as nn

# Residual block  残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 512全局平均池化自适应
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):  # blocks 代表残差执行的次数
        layer = []
        layer.append(ResidualBlock(in_channels, out_channels, stride))   # 第一个残差结构  输入通道和输出通道变化
        for i in range(1, blocks):
            layer.append(ResidualBlock(out_channels, out_channels))  # 后面的残差结构
        return nn.Sequential(*layer)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # out = self.canchakuai(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

model = ResNet18()

# # print(model.layer1)
# # print(model.layer1.parameters())
# # 权重的冻结  （先获取params 参数,再调节优化器）
# for param in model.layer1.parameters():
#     param.requires_grad = False
#
# # 通过优化器控制权重的更新
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#
#
#
# # for epoch in range(2):
# #     for i in dataloder():
# #         img, label = i
# #         img_out = model(img)
# #         loss_batch = loss(img_out, label)
# #
# #         optimizer.zero_grad()
# #         loss.backword()
# #         optimizer.step()
#
# # print(model.state_dict()['layer3.1.bn2.weight'])  # 获取权重信息
#
#
# # 解冻
#
# for name, parameter in model.layer1.named_parameters():
#     print(name)
#
#     print("=*5")
#     print(parameter)
#
#     split_name = name.split(".")[0]
#     if split_name in ["0"]:
#         parameter.requires_grad=True
#     else:
#         parameter.requires_grad=False
#
#     # 获取需要更新的参数
#     params = [p for p in model.parameters() if p.requires_grad]
#     # 设置好优化器
#     optimizer = torch.optim.SGD(params, lr=0.005,
#                                 momentum=0.9, weight_decay=0.005)
#
#     # 设置一个学习率的变化
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.33)
#
#     # 接着上次冻结的地方开始训练
#     num_epochs = 20
#     for epoch in range(2, num_epochs + 2, 1):
#
#         lr_scheduler.step()
#
#
#
#
#         # 模型的保存
#         if epoch > 10:
#             save_files = {
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_scheduler': lr_scheduler.state_dict(),
#                 'epoch': epoch
#             }
#             torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))
#
#     # model.eval()
#     # x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
#     # predictions = model(x)
#     # print(predictions)
#
#
#         # 模型的加载
#         if args.resume != "":
#             checkoint = torch.load(args.resume, map_location="cpu")
#             model.load_state_dict(checkpoint['model'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#             args.start_epoch = checkpoint['epoch'] + 1
#
#             if args.amp and "scaler" in checkpoint:
#                 scaler.load_state_dict(checkpoint["scaler"])
#
#         for epoch in range(args.start_epoch, args.epochs):
#
#             # 模型保存
#             save_files = {
#                 "model": model.state_dict(),
#                 "optimizer":optimizer.state_dict(),
#                 "lr_scheduler":lr_scheduler.state_dict()
#                 "epoch": epoch
#             }
#             if args.amp:
#                 save_files["scaler"] = scaler.state_dict()
#             torch.save("save_files", "{}.pth".format(epoch))

if __name__ == '__main__':
    # main()

    # scaler = torch.cuda.amp.GradScaler() if args.amp else None  # amp
    random_seed = 123
    torch.manual_seed(random_seed)

    a = torch.randn(1, 3, 112, 112)
    print(model(a))
