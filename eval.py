
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
        dataset = StsqDB(data_file='data/sampling/trim/three/seq_length_{}/val_split_1.pkl'.format(int(seq_length)),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif turn == True:
        dataset = StsqDB(data_file='data/sampling/trim/turn/seq_length_{}/val_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif step == True:
        dataset = StsqDB(data_file='data/sampling/trim/step/seq_length_{}/val_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif five == True:
        dataset = StsqDB(data_file='data/sampling/five/seq_length_{}/val_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif two == True:
        dataset = StsqDB(data_file='data/sampling/turn_step/seq_length_{}/val_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif four == True:
        dataset = StsqDB(data_file='data/sampling/four_one_half_rotate/seq_length_{}/val_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    else:
        dataset = StsqDB(data_file='data/sampling/seq_length_{}/val_split_1.pkl'.format(args.seq_length),
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
        element_preds = np.zeros(3)
        element_sum = np.zeros(3)
        confusion_matrix = np.zeros([3,3], int)
    elif turn== True:
        element_preds = np.zeros(6)
        element_sum = np.zeros(6)
        confusion_matrix = np.zeros([6,6], int)
    elif step== True:
        element_preds = np.zeros(6)
        element_sum = np.zeros(6)
        confusion_matrix = np.zeros([6,6], int)
    elif five== True:
        element_preds = np.zeros(5)
        element_sum = np.zeros(5)
        confusion_matrix = np.zeros([5,5], int)
    elif two== True:
        element_preds = np.zeros(2)
        element_sum = np.zeros(2)
        confusion_matrix = np.zeros([2,2], int)
    elif four== True:
        element_preds = np.zeros(4)
        element_sum = np.zeros(4)
        confusion_matrix = np.zeros([4,4], int)
    else:
        element_preds = np.zeros(12)
        element_sum = np.zeros(13)
        confusion_matrix = np.zeros([12,12], int)
        correct_label = np.zeros(13)



    for i, sample in enumerate(data_loader):
        images, labels = sample['images'].to(device), sample['labels'].to(device)
        logits = model(images)
        probs = F.softmax(logits.data, dim=1)  ##確率
        labels = labels.view(int(bs)*int(seq_length))
        if three == True:
            _, c, pred_label,v_label, conf, c_label= correct_preds(probs, labels.squeeze(),three, turn, step, five,two,four)
        elif turn == True:
            _, c, pred_label,v_label, conf, c_label= correct_preds(probs, labels.squeeze(),three, turn, step, five,two,four)
        elif step == True:
            _, c, pred_label,v_label, conf , c_label= correct_preds(probs, labels.squeeze(),three, turn, step, five,two,four)
        elif five == True:
            _, c, pred_label,v_label, conf , c_label= correct_preds(probs, labels.squeeze(),three, turn, step, five,two,four)
        elif two == True:
            _, c, pred_label,v_label, conf , c_label= correct_preds(probs, labels.squeeze(),three, turn, step, five,two,four)
        elif four == True:
            _, c, pred_label,v_label, conf , c_label= correct_preds(probs, labels.squeeze(),three, turn, step, five,two,four)
        else:
            _, c, pred_label,v_label, conf , c_label= correct_preds(probs, labels.squeeze(),three, turn, step, five,two,four)

        if disp:
            print(i, c,pred_label,v_label)
        correct.append(c)
        element_preds[int(pred_label)] += 1
        element_sum[int(v_label)] += 1
        correct_label[int(c_label)] +=1
        # for j in range(len(element_c)):
        #     element_correct[j].append(element_c[j])
        # for j in range(len(element_s)):
        #     element_sum[j].append(element_s[j])
        confusion_matrix = confusion_matrix + conf


    PCE = np.mean(correct)
    # all_element_correct = np.sum(element_correct, axis=1)
    all_element_correct =correct_label

    # all_element_sum = np.sum(element_sum, axis=1)
    all_element_sum = element_sum

    element_PCE = all_element_correct / all_element_sum
    return PCE, element_PCE, all_element_correct, all_element_sum, confusion_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=1)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--seq_length', default=300)
    parser.add_argument('--model_num', default=900)
    parser.add_argument('--three', action='store_true')
    parser.add_argument('--turn', action='store_true')
    parser.add_argument('--step', action='store_true')
    parser.add_argument('--five', action='store_true')
    parser.add_argument('--two', action='store_true')
    parser.add_argument('--four', action='store_true')

    args = parser.parse_args()


    split = args.split
    seq_length = args.seq_length
    n_cpu = 0
    bs = args.batch_size

    three = args.three
    turn = args.turn
    step = args.step
    five = args.five
    two = args.two
    four = args.four

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False,
                          three=three,
                          turn=turn,
                          step=step,
                          five= five,
                          two = two,
                          four = four)

    if three == True:
        save_dict = torch.load('models/sampling/trim/three/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))
    elif turn == True:
        save_dict = torch.load('models/sampling/trim/turn/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))
    elif step == True:
        save_dict = torch.load('models/sampling/trim/step/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))
    elif five == True:
        save_dict = torch.load('models/sampling/five/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))
    elif two == True:
        save_dict = torch.load('models/sampling/turn_step/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))
    elif four== True:
        save_dict = torch.load('models/sampling/four_one_half_rotate/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))
    else:
        save_dict = torch.load('models/sampling/seq_length_{}/swingnet_{}.pth.tar'.format(int(seq_length), args.model_num))


    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE, element_PCE, all_element_correct, all_element_sum, confusion_matrix = eval(model, split, int(seq_length), bs, n_cpu, True)
    print('Average PCE: {}'.format(PCE))

    if three == True:
        element_names = ['Step', 'Turn','No_element']
    elif turn == True:
        element_names = ['Bracket', 'Counter_turn', 'Loop', 'Rocker_turn', 'Three_turn', 'Twizzle']
    elif step == True:
        element_names = ['Change_edge', 'Chasse', 'Choctaw', 'Cross_roll', 'Mohawk', 'Toe_step']
    elif five == True:
        element_names = ['one_half_turn', 'one_step', 'both_half_turn', 'both_step', 'one_turn']
    elif two == True:
        element_names = [ 'Turn', 'Step',]
    elif four == True:
        element_names =  ['Bracket',  'Counter_turn','Rocker_turn', 'Three_turn']
    else:
        element_names =  ['Bracket', 'Change_edge', 'Chasse', 'Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','Miss']



    for j in range(len(element_PCE)):
        element_name = element_names[j]
        print('{}: {}  ({} / {})'.format(element_name, element_PCE[j], all_element_correct[j], all_element_sum[j]))


    ####################################################################
    print(confusion_matrix)
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=6, cmap=plt.get_cmap('Blues'))
    if args.three == True:
        plt.ylabel('Actual Category')
        plt.yticks(range(3), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(3), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'three_trim_same_frames_43.png')
    elif args.turn == True:
        plt.ylabel('Actual Category')
        plt.yticks(range(6), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(6), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'turn_trim_same_frames_43.png')
    elif args.step == True:
        plt.ylabel('Actual Category')
        plt.yticks(range(6), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(6), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'step_trim_same_frames_43.png')
    elif args.five == True:
        plt.ylabel('Actual Category')
        plt.yticks(range(5), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(5), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'five_same_frames_43.png')

    elif args.two == True:
        plt.ylabel('Actual Category')
        plt.yticks(range(2), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(2), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'sampling_ts.png')

    elif args.four == True:
        plt.ylabel('Actual Category')
        plt.yticks(range(4), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(4), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'four_one_half_rotate_same_frames_43.png')

    else:
        plt.ylabel('Actual Category')
        plt.yticks(range(12), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(12), element_names)

        save_dir = '/home/akiho/projects/golfdb/'
        plt.savefig(save_dir + 'sampling_19.png')