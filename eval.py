from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import StsqDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(model, split, seq_length, bs, n_cpu, disp):
    

    dataset = StsqDB(data_file='val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_40/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    element_correct = [ [] for i in range(13) ]

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'].to(device), sample['labels'].to(device)
        logits = model(images)       
        probs = F.softmax(logits.data, dim=1).view(bs*seq_length, -1)    
        labels = labels.view(bs*seq_length)
        _, _, _, _, c, element_c = correct_preds(probs, labels.squeeze())

        if disp:
            print(i, c)
        correct.append(c)
        for j in range(len(element_c)):
            element_correct[j].append(element_c[j])
        
        # images, labels = sample['images'], sample['labels']
        # # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        # batch = 0
        # while batch * seq_length < images.shape[1]:
        #     if (batch + 1) * seq_length > images.shape[1]:
        #         image_batch = images[:, batch * seq_length:, :, :, :]
        #     else:
        #         image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
        #     logits = model(image_batch.to(device))

        #     if batch == 0:
        #         probs = F.softmax(logits.data, dim=1).to(device).numpy()
        #     else:
        #         probs = np.append(probs, F.softmax(logits.data, dim=1).to(device).numpy(), 0)
        #     batch += 1
        # _, _, _, _, c = correct_preds(probs, labels.squeeze())
        # if disp:
        #     print(i, c)
        # correct.append(c)
    PCE = np.mean(correct)
    element_PCE = [ np.mean(element_c) for element_correct in element_c ]
    return PCE, element_PCE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', default=1)
    parser.add_argument('batch_size', default=16)
    parser.add_argument('seq_length', default=300) 
    args = parser.parse_args() 


    split = args.split
    seq_length = args.seq_length
    n_cpu = 6
    bs = args.batch_size

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/swingnet_800.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE, element_PCE = eval(model, split, seq_length, bs, n_cpu, True)
    print('Average PCE: {}'.format(PCE))

    # TODO: 
    element_names = ['Bracket', 'Change_edge', 'Chasse','Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','No_element']


    for j in range(len(element_PCE)):
        element_name = element_names[j]
        print('{}: {}'.format(element_name, element_PCE[j]))
