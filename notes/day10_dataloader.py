import torch
from torch.utils.data import Dataset, DataLoader        #导入 PyTorch 提供的 Dataset 基类和 DataLoader


class SimpleDataset(Dataset):       #定义了一个名为 SimpleDataset 的类，并继承 Dataset
    def __init__(self):     #初始化方法，定义数据集的特征和标签
        # 8个样本，每个样本有2个特征
        self.features = torch.tensor(       #self 表示当前创建的数据集对象
            [
                [1.0, 2.0],
                [2.0, 1.0],
                [3.0, 3.5],
                [4.0, 2.0],
                [5.0, 4.5],
                [6.0, 5.0],
                [7.0, 3.0],
                [8.0, 6.0],
            ],
            dtype=torch.float32
        )

        # 每个样本对应的类别标签
        self.labels = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1],
            dtype=torch.long        # 标签数据类型为长整型（int64）
        )

    # 返回数据集的样本数
    def __len__(self):
        return len(self.features)

    # 根据索引返回对应的特征和标签
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


#主程序入口
if __name__ == "__main__":
    dataset = SimpleDataset()

    print("数据集样本数：", len(dataset))
    print("第一个样本：", dataset[0])
    print("最后一个样本：", dataset[-1])

    # #使用 DataLoader 进行分批加载数据
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=3,       #每个 batch 的样本数为 3
    #     shuffle=False       #不打乱数据
    # )

    # print("\n开始分批遍历：")

    # for batch_index, (batch_features, batch_labels) in enumerate(
    #     dataloader,
    #     start=1
    # ):      #enumerate() 函数用于在遍历 dataloader 时获取批次索引和对应的特征、标签
    #     print(f"\n第 {batch_index} 个 batch")
    #     print("特征：")
    #     print(batch_features)
    #     print("标签：", batch_labels)
    #     print("特征形状：", batch_features.shape)
    #     print("标签形状：", batch_labels.shape)

    #连续训练两轮，每轮打印所有batch，并将shuffle设置为True
    dataloader=DataLoader(
        dataset,
        batch_size=3,
        shuffle=True       #每轮训练打乱数据顺序
    )

    for epoch in range(2):
        print(f"\n第 {epoch+1} 轮训练：")

        for batch_index,(batch_features,batch_labels) in enumerate(
            dataloader,
            start=1
        ):
            print(f"\n第 {batch_index} 个 batch")
            print("特征：")
            print(batch_features)
            print("标签：", batch_labels)
            print("特征形状：", batch_features.shape)
            print("标签形状：", batch_labels.shape)
