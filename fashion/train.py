import torch
from models import Classifier
from torch import nn, optim
import matplotlib.pyplot as plt
from datasets import load_dataset

class training():
    def __init__(self):
        self.epoch = 15

    def training(self):
        # best_model用于保存测试效果最好的模型参数，best_loss用于保存效果最好的模型测试loss
        best_model = {}
        best_loss = 1
        # 初始化模型
        model = Classifier()
        # 创建训练数据和测试数据加载器
        trainloader, testloader = load_dataset()
        # 定义loss
        loss_function = nn.NLLLoss()
        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        # train_losses和test_losses分别用于记录训练过程中每个epoch的loss
        train_losses, test_losses = [], []
        print("------------开始训练------------")
        # 开始训练
        for i in range(self.epoch):
            running_loss = 0
            # 导入一个批次的数据
            for images, labels in trainloader:
                # 梯度清零
                optimizer.zero_grad()
                # 推理
                predicts = model(images)
                # 计算loss
                loss = loss_function(predicts, labels)
                # 反向传播
                loss.backward()
                # 优化参数
                optimizer.step()
                # 将张量转化为标量 方便累加
                running_loss += loss.item()

            else:
                test_loss = 0
                # 准确度
                accuracy = 0
                with torch.no_grad():
                    # 使模型进入评估模式，例如取消dropout
                    model.eval()

                    for images, labels in testloader:
                        # 推理
                        predicts = model(images)
                        test_loss += loss_function(predicts, labels)
                        # 计算每种类别的概率 最大值为1
                        pr = torch.exp(predicts)
                        # 取最大的概率和类别
                        top_p, top_class = pr.topk(1, dim=1)
                        # 转化为0,1，方便计算准确度
                        res = top_class == labels.view(*top_class.shape)

                        # 计算准确度
                        accuracy += torch.mean(res.type(torch.FloatTensor))
                    # 使模型进入训练模式
                    model.train()
                    # 计算当前epoch的loss
                    epoch_train_loss = running_loss/len(trainloader)
                    epoch_test_loss = test_loss/len(testloader)
                    epoch_accuracy = accuracy/len(testloader)
                    # 记录所有epoch的loss
                    train_losses.append(epoch_train_loss)
                    test_losses.append(epoch_test_loss)
                    # 保存最佳模型
                    if epoch_test_loss < best_loss:
                        best_model = model.state_dict()
                        best_loss = epoch_test_loss
                    print("epoch:{}/{}..".format(i+1, self.epoch),
                          "epoch_train_loss:{}..".format(epoch_train_loss),
                          "epoch_test_loss:{}..".format(epoch_test_loss),
                          "epoch_accuracy:{}".format(epoch_accuracy))
        torch.save(best_model, "saved_models/best_model.pth")
        plt.plot(train_losses, label="train_losses")
        plt.plot(test_losses, label="test_losses")
        plt.legend()
        plt.show()





