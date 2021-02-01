from comet_ml import Experiment
from tqdm import tqdm
from dataloader import StsqDB, Normalize, ToTensor
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=1)
    parser.add_argument('--iteration', default=8000)
    parser.add_argument('--it_save', default=100)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--seq_length', default=300)
    parser.add_argument('--three', action='store_true')
    parser.add_argument('--turn', action='store_true')
    parser.add_argument('--step', action='store_true')
    parser.add_argument('--five', action='store_true')
    parser.add_argument('--two', action='store_true')
    parser.add_argument('--four', action='store_true')
    args = parser.parse_args()
    # これ以降、このファイル内では "args.iterration" で2000とか呼び出せるようになる

    experiment = Experiment(api_key='d7Xjw6KSK6KL7pUOhXJvONq9j', project_name='stsqdb')
    hyper_params = {
    'batch_size': args.batch_size,
    'iterations' : args.iteration,
    'seq_length' : args.seq_length,
    'three' : args.three,
    }

    experiment.log_parameters(hyper_params)

    # training configuration
    split = args.split
    iterations = args.iteration
    it_save = args.it_save # save model every 100 iterations
    n_cpu = 6
    seq_length = args.seq_length
    bs = args.batch_size  # batch size
    k = 10  # frozen layers

    three = args.three
    turn = args.turn
    step = args.step
    five = args.five
    two = args.two
    four = args.four


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Load Model')

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False,
                          three=three,
                          turn = turn,
                          step =step,
                          five = five,
                          two = two,
                          four = four
                          )
    #print('model.py, class EventDetector()')

    freeze_layers(k, model)
    #print('utils.py, func freeze_laters()')
    model.train()
    model.to(device)
    print('Loading Data')


    # TODO: vid_dirのpathをかえる。stsqの動画を切り出したimage全部が含まれているdirにする
    if three == True:
        dataset = StsqDB(data_file='data/sampling/trim/three/seq_length_{}/train_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif turn == True:
        dataset = StsqDB(data_file='data/sampling/trim/turn/seq_length_{}/train_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif step == True:
        dataset = StsqDB(data_file='data/sampling/trim/step/seq_length_{}/train_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif five == True:
        dataset = StsqDB(data_file='data/sampling/five/seq_length_{}/train_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)

    elif two == True:
        dataset = StsqDB(data_file='data/sampling/two/turn_step/seq_length_{}/train_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    elif four == True:
        dataset = StsqDB(data_file='data/sampling/four_one_half_rotate/seq_length_{}/train_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)
    else:
        dataset = StsqDB(data_file='data/sampling/seq_length_{}/train_split_1.pkl'.format(args.seq_length),
                    vid_dir='data/videos_56/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)

    print('dataloader.py, class StsqDB()')
    # dataset.__len__() : 1050


    data_loader = DataLoader(dataset,
                             batch_size=int(bs),
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # dataset.__len__() : 47 (dataset/bs)


    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    # TODO: edit weights shape from golf-8-element to stsq-12-element
    if three == True:
        weights = torch.FloatTensor([1, 1, 1]).to(device)
    elif turn == True:
        weights = torch.FloatTensor([1, 1, 1,1,1,1]).to(device)
    elif step == True:
        weights = torch.FloatTensor([1, 1, 1,1,1,1]).to(device)
    elif five == True:
        weights = torch.FloatTensor([1, 1, 1,1,1]).to(device)
    elif two == True:
        weights = torch.FloatTensor([1, 1]).to(device)
    elif four == True:
        weights = torch.FloatTensor([1, 1, 1, 1]).to(device)
    else:
        weights = torch.FloatTensor([1, 1, 1, 1,1,1,1,1,1,1,1,1]).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)  ##lambda:無名関数

    losses = AverageMeter()
    #print('utils.py, class AverageMeter()')

    if not os.path.exists('models/sampling/'):
        os.mkdir('models/sampling/')



    epoch = 0
    for epoch in range(int(iterations)):
        print(epoch)

        for sample in tqdm(data_loader):
            images, labels = sample['images'].to(device), sample['labels'].to(device)
            logits= model(images.float())
            labels = labels.view(int(bs)*int(seq_length))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()


            print('tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

        if three == True:
            epoch += 1
            if epoch % int(it_save) == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/sampling/trim/three/seq_length_{}/swingnet_{}.pth.tar'.format(args.seq_length, epoch))
        elif turn == True:
            epoch += 1
            if epoch % int(it_save) == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/sampling/trim/turn/seq_length_{}/swingnet_{}.pth.tar'.format(args.seq_length, epoch))
        elif step == True:
            epoch += 1
            if epoch % int(it_save) == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/sampling/trim/step/seq_length_{}/swingnet_{}.pth.tar'.format(args.seq_length, epoch))
        elif five == True:
            epoch += 1
            if epoch % int(it_save) == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/sampling/five/seq_length_{}/swingnet_{}.pth.tar'.format(args.seq_length, epoch))
        elif two == True:
            epoch += 1
            if epoch % int(it_save) == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/sampling/turn_step/seq_length_{}/swingnet_{}.pth.tar'.format(args.seq_length, epoch))
        elif four == True:
            epoch += 1
            if epoch % int(it_save) == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/sampling/four_one_half_rotate/seq_length_{}/swingnet_{}.pth.tar'.format(args.seq_length, epoch))
        else:
            epoch += 1
            if epoch % int(it_save) == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/sampling/seq_length_{}/swingnet_{}.pth.tar'.format(args.seq_length, epoch))

            if epoch == iterations:
                break

        experiment.log_parameter("train_loss", loss.item(), step=epoch)