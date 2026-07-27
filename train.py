import torch
from torch import nn

from data.fashion_mnist import create_dataloaders, create_datasets
from models.mlp import MLP
from utils.engine import evaluate, train_one_epoch
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    # TODO 1：固定随机种子
    torch.manual_seed(42)

    #当电脑可以使用 CUDA 显卡时，给所有 GPU 的随机数生成器设置固定种子 42
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # TODO 2：选择 CPU 或 GPU
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("使用设备：",device)

    # TODO 3：创建 Dataset 和 DataLoader
    train_dataset,test_dataset=create_datasets()

    train_dataloader,test_dataloader=create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=64
    )

    # TODO 4：创建模型、损失函数和优化器
    model=MLP().to(device)
    loss_fn=nn.CrossEntropyLoss()

    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=0.1
    )

    # TODO 5：训练 3 个 epoch，并在每轮后进行验证和打印指标
    num_epochs=3

    results_dir = Path(__file__).resolve().parent / "results" / "fashion"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = results_dir / "best_mlp.pth"
    curve_path = results_dir / "training_curves.png"

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    best_test_accuracy = 0.0

    for epoch in range(1,num_epochs+1):
        train_loss,train_accuracy=train_one_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_dataloader,
            loss_fn,
            device
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

            #把模型当前学到的参数保存到 best_model_path 指定的文件中
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.2%} | "
            f"test loss: {test_loss:.4f} | "
            f"test acc: {test_accuracy:.2%}"
        )

    #绘制曲线
    epochs = range(1, num_epochs + 1)

    #创建一张画布，在画布中横向放置两个子图
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    #axes[0]表示左边的第一个子图对象
    axes[0].plot(epochs, train_losses, marker="o", label="Train")
    axes[0].plot(epochs, test_losses, marker="o", label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    #axes[1]表示右边的第二个子图对象
    axes[1].plot(epochs, train_accuracies, marker="o", label="Train")
    axes[1].plot(epochs, test_accuracies, marker="o", label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)

    print("最佳模型已保存到：", best_model_path)
    print("训练曲线已保存到：", curve_path)

    #重新加载最佳模型
    loaded_model = MLP().to(device)

    #从 best_model_path 指定的文件中读取模型参数，并把读取结果保存到变量 state_dict 中
    state_dict = torch.load(        #从磁盘文件中读取之前用 torch.save() 保存的对象
        best_model_path,
        map_location=device,        #把文件中的张量加载到 device 指定的设备上
        weights_only=True
    )

    loaded_model.load_state_dict(state_dict)

    loaded_test_loss, loaded_test_accuracy = evaluate(
        loaded_model,
        test_dataloader,
        loss_fn,
        device
    )

    print(
        f"重新加载后的模型 | "
        f"test loss: {loaded_test_loss:.4f} | "
        f"test acc: {loaded_test_accuracy:.2%}"
    )

if __name__ == "__main__":
    main()
