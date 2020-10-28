import torch.nn as nn


def weights_init(model, method=nn.init.xavier_normal_):
    """Weights init for pytorch model or torch.nn.Sequential object.

    Args:
        model ([type]): [description]
        method ([type], optional): [description]. Defaults to nn.init.xavier_normal_.

    Returns:
        model: initialized model
    """
    def fun(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            method(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(fun)
    return model