# pylint: disable=import-error
"""
This module demonstrates basic operations with PyTorch tensors,
including creation, basic arithmetic, matrix multiplication,
and automatic differentiation.
"""

import torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor x:", x)

# Perform a basic operation
y = x + 2
print("Tensor y (x + 2):", y)

# Perform matrix multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.matmul(a, b)
print("Matrix multiplication result:\n", c)

# Use autograd for automatic differentiation
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.mean()
z.backward()
print("Gradient of x:", x.grad)
