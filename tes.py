import torch
print('version', torch.backends.cudnn.version())
print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
