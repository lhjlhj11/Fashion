import torch
from torchvision import datasets, transforms

def load_dataset():
    # 数据预处理流程 将数据转化为张量并对其进行归一化
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    # 下载训练数据并创建训练数据加载器
    train_set = datasets.FashionMNIST("dataset/", download=True, train=True, transform=data_transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    # 下载测试数据并创建测试数据加载器
    test_set = datasets.FashionMNIST("dataset/", download=True, train=False, transform=data_transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    # 返回训练数据和测试数据加载器
    return trainloader, testloader




