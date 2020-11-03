import torch.nn as nn
from torch.utils.data import Dataset


def weights_init(model, method='xavier'):
    """Weights init inplace for pytorch model or torch.nn.Sequential object. default is xavier init.

    Args:
        model ([type]): [description]
        method ([type], optional): [description]. Defaults to nn.init.xavier_normal_.

    Returns:
        model: initialized model
    """
    if method == 'xavier':
        init_fun = nn.init.xavier_normal_
    elif method == 'kaiming':
        init_fun = nn.init.kaiming_normal_
    else:
        print('no init is applied')
        return

    def fun(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_fun(m.weight)
            if not m.bias is None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)

    model.apply(fun)


class TensorDataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, X, y=None, transforms=None):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.
            transforms ([type], optional): [description]. Defaults to None.
        """
        self.X = X
        self.transforms = transforms
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = self.X[index]
        if self.transforms:
            sample = self.transforms(sample)
        if self.y is not None:
            return sample, self.y[index]
        else:
            return sample
