import torch
import torch.nn as nn
from torch.utils.data import Dataset
import logging


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


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels, loader, transform=None):
        """[summary]

        Args:
            paths ([type]): [description]
            labels ([type]): [description]
            loader ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
        """
        self.image_paths = paths
        self.label = labels
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        img = self.loader(self.image_paths[index])
        label = self.label[index]
        if isinstance(self.transform, tuple):
            if label == 0:
                img = self.transform[1](img)
            else:
                img = self.transform[0](img)

        elif self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.label)


class data_prefetcher():
    def __init__(self, loader, mean=None, std=None):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = mean
        self.std = std
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
            self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            if self.mean is not None and self.std is not None:
                self.next_input = self.next_input.sub_(self.mean).div_(
                    self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k
    Used for torch tensor as input.
    Args:
        output ([type]): [description]
        target ([type]): [description]
        topk (tuple, optional): [description]. Defaults to (1, ).

    Returns:
        list: [description]
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res