import torch
import torch.nn.functional as F  # 使用relu()作为激活函数
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split as tts

data = torch.load("data.pth")
label = data["label"]
feature = data["feature"]

X_train, X_test, y_train, y_test = tts(feature, label, test_size=0.2, random_state=0)

from my_network import MyClassifier

model = MyClassifier()

criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)  # momentum：带冲量的优化算法
optimizer = optim.Adam(model.parameters(), lr=0.0002)

def train(epoch):
    running_loss = 0.0
    for batch_idx in range(len(X_train)):
        inputs, target = X_train[batch_idx], y_train[batch_idx]
        optimizer.zero_grad()  # 优化器清零

        outputs = model(inputs)  # 获得预测值，forward
        loss = criterion(outputs, target)  # 获得损失值
        loss.backward()  # backward
        optimizer.step()  # update

        running_loss += loss.item()  # 注意加上item：不构成计算图
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for i in range(len(X_test)):
            images, labels = X_test[i], y_test[i]
            outputs = model(images)
            outputs = outputs.numpy()
            predicted = np.argmax(outputs)
            total += 1  # 测试了多少个数据
            correct += (predicted == labels).sum().item()  # 计算有多少个预测正确
    print('Accuuracy on test set: %d %%' % (100 * correct / total))  # 输出正确率


if __name__ == '__main__':
    op1 = input('key: ')
    if op1 == 't':
        for epoch in range(10):
            train(epoch)
            # torch.save(model,
            #            'ano_model.pth')
            # model = torch.load('ano_model.pth')
            test()
        # torch.save(model, 'myModel2.pth')
        op2 = input('key: ')
        if op2 == 's':
            torch.save(model.state_dict(), 'myModel4.pth')
    else:
        model = torch.load('AIProject3/myModel.pth')
        test()
        torch.save(model.state_dict(), 'myModel4.pth')
    