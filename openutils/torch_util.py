import torch.nn as nn


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
