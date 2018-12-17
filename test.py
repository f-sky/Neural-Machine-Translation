import torch

if __name__ == '__main__':
    a = torch.rand((2, 30, 1))
    print(a.requires_grad)
