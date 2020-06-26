import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def correct_preds(probs, labels, tol=-1):

    labels = labels.to('cpu').detach().numpy().copy()  ##labelsをnumpyに変換
    preds= np.zeros(len(labels))
    correct = []
    correct_labels = []

    for i in range(len(labels)):
        preds[i] = np.argsort(probs[i,:])[-1] 

    for i in range(len(labels)):
        if labels[i] == preds[i]:
            correct.append(1)
            correct_labels.append(labels[i])
        else:
            correct.append(0)

    return labels, correct_labels, correct


def freeze_layers(num_freeze, net):
    # print("Freezing {:2d} layers".format(num_freeze))
    i = 1
    for child in net.children():
        if i ==1:
            j = 1
            for child_child in child.children():
                if j <= num_freeze:
                    for param in child_child.parameters():
                        param.requires_grad = False
                j += 1
        i += 1