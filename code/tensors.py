import torch
import numpy as np

# 直接初始化一个tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 从Numpy数组中初始化
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 从其他tensor中初始化
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# 使用随机数和常数初始化
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 获取tensor的属性
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 如果有的话，我们把我们的tensor移到GPU上
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

tensor = torch.ones(4, 4)
print('第一行: ', tensor[0])
print('第一列：', tensor[:, 0])
print('最后一列：', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 算术运算
# 这将计算两个tensor之间的矩阵乘法，y1, y2, y3将有相同的值
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# 这将计算出元素相乘的结果。z1，z2, z3有相同的值
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 单元素tensor:如果你有一个单元素tensor，例如将一个tensor的所有值总计成一个值，你可以使用item()将其变换为Python数值。
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


