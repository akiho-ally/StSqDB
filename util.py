import numpy as np

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

    preds = np.zeros(len(labels))
    correct = []
    each_element_sum = np.zeros(13)
    each_element_preds = np.zeros(13)

    for i in range(len(labels)):
        preds[i] = np.argsort(probs[i,:])[-1] 

    for i in range(len(labels)):
        if labels[i] == preds[i]:
            correct.append(1)
        else:
            correct.append(0)

    for i in range(len(labels)):
        label_id = labels[i]
        each_element_sum[int(label_id)] += 1
        if correct[i] == 1:
            each_element_preds[int(label_id)] += 1

        
    return preds, correct , each_element_preds, each_element_sum


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