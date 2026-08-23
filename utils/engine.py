import torch


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """完整遍历一次训练集，返回平均损失和准确率。"""
    model.train()

    total_loss = 0.0
    total_correct = 0  # 记录预测正确数量
    total_samples = 0  # 记录总样本数量

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        logits = model(images)

        loss = loss_fn(logits, labels)  # 一批数据的平均损失
        loss.backward()

        # 根据反向传播计算出的梯度，修改模型里的 weight 和 bias，让下一次预测更准确
        optimizer.step()

        batch_size = labels.size(0)  # 获取 labels 第 0 维的长度
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss,accuracy


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """完整遍历一次验证集，返回平均损失和准确率。"""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy
