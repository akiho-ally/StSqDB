import numpy as np
import collections
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





def correct_preds(probs, labels, three, turn, step, five, two, four, tol=-1):


    # events, _  = np.where(labels < 13)
    # preds = np.zeros(len(events))
    # each_element_preds = np.zeros(13)

    # if tol == -1:  ##許容誤差
    #     #tol = int(max(np.round((events[5] - events[0])/30), 1))  ##(impact-address)/fps
    #     tol = 5

    # for i in range(len(events)):
    #     preds[i] = np.argsort(probs[i,:])[-1]  ##probsのi列目をsortしたものの一番大きいインデックス？？  ##probs.shape:(300,13)

    # deltas = np.abs(events-preds)  ##abs:絶対値
    # correct = (deltas <= tol).astype(np.uint8)  #deltaが誤差以下なら1,誤差以上なら0

    # for i in range(len(events)):
    #     label_id = events[i]
    #     if correct[i] == 1:
    #         each_element_preds[int(label_id)] += 1


    # return events, preds, deltas, tol, correct, each_element_preds


    preds = np.zeros(len(labels))
    # correct = []


    if three == True:
        each_element_sum = np.zeros(3)
        each_element_preds = np.zeros(3)
        confusion_matrix = np.zeros([3,3], int)
        c_num  = np.zeros(3)
    elif turn == True:
        each_element_sum = np.zeros(6)
        each_element_preds = np.zeros(6)
        confusion_matrix = np.zeros([6,6], int)
        c_num  = np.zeros(6)
    elif step == True:
        each_element_sum = np.zeros(6)
        each_element_preds = np.zeros(6)
        confusion_matrix = np.zeros([6,6], int)
        c_num  = np.zeros(6)
    elif five == True:
        each_element_sum = np.zeros(5)
        each_element_preds = np.zeros(5)
        confusion_matrix = np.zeros([5,5], int)
        c_num  = np.zeros(5)
    elif two == True:
        each_element_sum = np.zeros(2)
        each_element_preds = np.zeros(2)
        confusion_matrix = np.zeros([2,2], int)
        c_num  = np.zeros(2)
    elif four == True:
        each_element_sum = np.zeros(4)
        each_element_preds = np.zeros(4)
        confusion_matrix = np.zeros([4,4], int)
        c_num  = np.zeros(4)
    else:
        each_element_sum = np.zeros(12)
        each_element_preds = np.zeros(12)
        confusion_matrix = np.zeros([12,12], int)
        element_num  = 12

    for i in range(len(labels)):
        preds[i] = np.argsort(probs[i,:])[-1]
        # confusion_matrix[labels[i].item(), int(preds[i].item())] += 1


    # for i in range(len(labels)):
    #     if labels[i] == preds[i]:
    #         correct.append(1)
    #     else:
    #         correct.append(0)

    preds_count = collections.Counter(preds)
    tuple_preds_count = preds_count.most_common()
    labels_count = collections.Counter(labels)
    tuple_labels_count = labels_count.most_common()

    pred_video_label = tuple_preds_count[0][0]
    video_label = tuple_labels_count[0][0]


    if pred_video_label == video_label:
        correct = 1
        c_label = video_label
    else:
        correct = 0
        c_label = element_num

    confusion_matrix[video_label, int(pred_video_label)] += 1


    # if correct.count(1) >= len(correct)/2:
    #     pred_vide0_label = labels[0]
    # else:

    #     pred_video_label =

    # for i in range(len(labels)):
    #     label_id = labels[i]
    #     each_element_sum[int(label_id)] += 1
    #     if correct[i] == 1:
    #         each_element_preds[int(label_id)] += 1

    # import pdb; pdb.set_trace()

    return preds, correct , pred_video_label, video_label, confusion_matrix, c_label


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