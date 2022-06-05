import torch

x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

print(x)
expand_size = 3
expanded_return_idx = (
            torch.arange(x.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1)
        )

print(expanded_return_idx)

input_ids = x.index_select(0, expanded_return_idx)
print(input_ids)