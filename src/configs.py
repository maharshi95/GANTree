import torch


class Config:
    use_gpu = torch.cuda.is_available()
    base_port = 7001
