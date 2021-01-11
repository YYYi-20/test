import logging
import numbers

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class focal_loss(nn.Module):
    '''focal_loss func, L = -α(1-yi)**γ *ce_loss(xi,yi)
    '''
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow(
            (1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


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
        if self.transform is not None:
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


class ConfusionMeter():
    """Maintains a confusion matrix for a given calssification problem.
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not
    """
    def __init__(self, k, normalized=False, label_name=None):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.label_name = label_name
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def update(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted score btained
            from the model for N examples and K classes or an N-tensor of integer
            values between 0 and K-1. 
            target (tensor): Can be a N-tensor of integer values assumed to be
            integer values between 0 and K-1 or N x K tensor, where targets are
            assumed to be provided as one-hot vectors
        """
        predicted = predicted.cpu().data.numpy()  #make sure  no grad
        target = target.cpu().data.numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.k**2)
        assert bincount_2d.size == self.k**2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds 
            to ground-truth targets and columns corresponds to predicted targets.
        """
        self.p, self.r, self.f1 = confision_to_pre_recall_f1(self.conf)
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

    def __str__(self):
        value = self.value()
        columns = [f'label_{i}' for i in range(self.k)]
        index = columns + ['pre', 'recall', 'f1']
        value = np.vstack([value, self.p, self.r, self.f1])
        result = pd.DataFrame(data=value, index=index, columns=columns)
        return '\n' + str(result)


def _divide(a, b):
    return np.divide(a.astype('float32'),
                     b.astype('float32'),
                     out=np.zeros_like(a, dtype='float32'),
                     where=b != 0)


def confision_to_pre_recall_f1(confusion_matrix):
    ''' row is true, column is predicted
    '''
    diagonal = np.diagonal(confusion_matrix).flatten()
    col_sum = np.sum(confusion_matrix, axis=0).flatten()
    row_sum = np.sum(confusion_matrix, axis=1).flatten()

    p = _divide(diagonal, col_sum)
    r = _divide(diagonal, row_sum)
    f1 = _divide(2 * p * r, p + r)
    return p, r, f1


class AUCMeter():
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.
    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def update(self, output, target):
        if torch.is_tensor(output):
            # make sure no grad in output and target
            output = output.cpu().data.squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().data.squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)

    def value(self):
        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return (0.5, 0.0, 0.0)

        # sorting the arrays
        scores, sortind = torch.sort(torch.from_numpy(self.scores),
                                     dim=0,
                                     descending=True)
        scores = scores.numpy()
        sortind = sortind.numpy()

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        # calculating area under curve using trapezoidal rule
        n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return (area, tpr, fpr)


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
