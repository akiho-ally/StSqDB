from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import StsqDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import collections
import matplotlib.pyplot as plt

import argparse

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
def eval(model, split, seq_length, bs, n_cpu, disp):

    if three == True:
        dataset = StsqDB(data_file='data/sameframes/three/seq_length_{}/val_split_1.pkl'.format(int(seq_length)),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)

    data_loader = DataLoader(dataset,
                             batch_size=int(bs),
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=True)

    correct = []

    if three == True:
        element_correct = [ [] for i in range(3) ]
        element_sum = [ [] for i in range(3)]
        confusion_matrix = np.zeros([3,3], int)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'].to(device), sample['labels'].to(device)
        logits, _ = model(images)
        probs = F.softmax(logits.data, dim=1)  ##確率
        labels = labels.view(int(bs)*int(seq_length))
        _, c, element_c, element_s, conf = correct_preds(probs, labels.squeeze(),three)
        if disp:
            print(i, c)
        correct.append(c)
        for j in range(len(element_c)):
            element_correct[j].append(element_c[j])
        for j in range(len(element_s)):
            element_sum[j].append(element_s[j])
        confusion_matrix = confusion_matrix + conf


    PCE = np.mean(correct)
    all_element_correct = np.sum(element_correct, axis=1)
    all_element_sum = np.sum(element_sum, axis=1)
    element_PCE = all_element_correct / all_element_sum
    return PCE, element_PCE, all_element_correct, all_element_sum, confusion_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=1)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--seq_length', default=300)
    parser.add_argument('--model_num', default=900)
    parser.add_argument('--three', action='store_true')
    args = parser.parse_args()


    split = args.split
    seq_length = args.seq_length
    n_cpu = 0
    bs = args.batch_size

    three = args.three

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False,
                          three=three)

    save_dict = torch.load('models/sameframes/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE, element_PCE, all_element_correct, all_element_sum, confusion_matrix = eval(model, split, int(seq_length), bs, n_cpu, True)
    print('Average PCE: {}'.format(PCE))

    if three == True:
        element_names = ['Step', 'Turn','No_element']



    for j in range(len(element_PCE)):
        element_name = element_names[j]
        print('{}: {}  ({} / {})'.format(element_name, element_PCE[j], all_element_correct[j], all_element_sum[j]))


    ####################################################################
    print(confusion_matrix)
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=250, cmap=plt.get_cmap('Blues'))
    if args.three == True:
        plt.ylabel('Actual Category')
        plt.yticks(range(13), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(13), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'three_same_frames_43.png')