import torch

#每行代表一个学生，每列代表数学/英语/编程成绩
scores=torch.tensor(
    [
        [85,90,88],
        [78,86,92],
        [91,89,95]
    ],
    dtype=torch.float32
)

print("成绩 Tensor：")
print(scores)

print("形状：", scores.shape)
print("维数：", scores.ndim)
print("元素总数：", scores.numel())
print("数据类型：", scores.dtype)
print("所在设备：", scores.device)      #数据位于 CPU 还是 GPU

#第一名同学的所有成绩
print("第一名同学的所有成绩：", scores[0])

#所有学生的编程成绩
print("所有学生的编程成绩：", scores[:,2])

#前两名学生的英语和编程成绩
print("前两名学生的英语和编程成绩：", scores[0:2,1:3])

nums=torch.arange(1,13, dtype=torch.float32)
matrix=nums.reshape(3,4)
matrix_1=matrix.reshape(2,-1)  #-1表示自动计算维度

print("3×4 矩阵：")
print(matrix)
print("形状：", matrix.shape)

print("2×6 矩阵：")
print(matrix_1)
print("形状：", matrix_1.shape)

a = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    dtype=torch.float32
)

b = torch.tensor(
    [
        [1, 2],
        [3, 4],
        [5, 6]
    ],
    dtype=torch.float32
)

print(a*2)
c=a@b
print(c)
print(c.shape)
#预测c的形状为（2,2）

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)       #选择设备，如果有 GPU 则使用 GPU，否则使用 CPU

a_device = a.to(device)     #将张量 a 迁移到指定设备
b_device = b.to(device)

device_result = a_device @ b_device
cpu_result = device_result.cpu()        #将结果从 GPU 移回 CPU
print("选择的设备：", device)
print("原始 a 的设备：", a.device)
print("迁移后 a 的设备：", a_device.device)
print("GPU 计算结果的设备：", device_result.device)
print("移回后的设备：", cpu_result.device)
print("CPU/GPU 结果一致：", torch.allclose(c, cpu_result))