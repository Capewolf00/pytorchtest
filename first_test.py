# pylint: disable=import-error

import torch

shape = (2, 3, 5)
tensor = torch.rand(shape)

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(tensor)