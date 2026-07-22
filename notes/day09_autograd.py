import torch
from pathlib import Path
import matplotlib.pyplot as plt

x=torch.tensor(
    2.0,
    dtype=torch.float32,
    requires_grad=True
)

y=x**2+3*x+1
print("y:",y)
print("反向传播前的梯度：",x.grad)
y.backward()      #反向传播，计算梯度
print("反向传播后的梯度：",x.grad)
x.grad.zero_()      #梯度清零
z=x**3
z.backward()      #反向传播，计算梯度
print("z对x的梯度：",x.grad)        #.backward()负责计算梯度。梯度默认会累加，因此下一次求导前必须清零。

#学习优化器，用梯度下降拟合直线 \(y=2x+3\)
# 训练数据：实际规律是 y = 2x + 3
x_train = torch.tensor([-2., -1., 0., 1., 2.])
y_train = 2 * x_train + 3

# 模型参数：先故意从 0 开始
weight=torch.tensor(0.0,requires_grad=True)
bias=torch.tensor(0.0,requires_grad=True)

#随机梯度下降优化器
optimizer=torch.optim.SGD([weight,bias],lr=0.05)        #创建参数优化器，指定学习率
loss_history=[]        #记录每次迭代的损失值

#训练循环
for epoch in range(201):
    #1.清除上一轮梯度
    optimizer.zero_grad()      
    
    #2.前向传播计算预测值
    predictions=weight*x_train+bias
    
    #3.计算均方误差
    loss=((predictions-y_train)**2).mean()
    
    #4.计算参数梯度
    loss.backward()
    
    #5.更新参数
    optimizer.step()
    loss_history.append(loss.item())        #记录每次迭代的损失值,item()方法将张量转换为 Python 数值类型
    
    #打印训练信息,每 20 次迭代打印一次
    if epoch%20==0:
        print(
            f"epoch={epoch:3d},"
            f"loss={loss.item():.6f},"
            f"weight={weight.item():.4f},"
            f"bias={bias.item():.4f}"
        )

#.resolve()方法将相对路径转换为绝对路径，.parents[1]表示获取当前文件的上面第二级父目录，即CV-SUMMER-2026        
output_dir = Path(__file__).resolve().parents[1] / "results" / "week2"
#创建输出目录，如果目录已存在则不报错
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
#绘制损失曲线,loss_history是y轴，损失值,横坐标是迭代次数
plt.plot(loss_history, color="royalblue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Linear Regression Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(output_dir / "day09_loss_curve.png", dpi=150)
plt.show()
    
    

