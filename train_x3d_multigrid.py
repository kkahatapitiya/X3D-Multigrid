import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torchsummary import summary
#from thop import profile

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np
from barbar import Bar
from apmeter import APMeter

import x3d as resnet_x3d

from charades_x3d_multigrid_dataset import Charades as Dataset
from charades_x3d_multigrid_dataset import Charades as Dataset_Full

import cycle_batch_sampler as cbs
import dataloader as DL

import warnings
warnings.filterwarnings("ignore")

X3D_VERSION = 'M'

def setup_data(root, split, mode, batch_size, epochs, crop_size, resize_size, num_frames):

  # base BS of 64 -> ~250 iterations per epoch for kin
  steps_per_epoch = 8000/batch_size
  num_iterations = int(epochs * steps_per_epoch)
  schedule = [0, int(0.6*num_iterations), int(0.85*num_iterations), num_iterations]
  long_cycle = [8, 4, 2, 1]
  train_transforms = transforms.Compose([videotransforms.RandomHorizontalFlip()])

  dataset = Dataset(split, 'training', root, mode, train_transforms, resize_size=resize_size,
                          crop_size=crop_size, center_crop=False, num_f=num_frames)

  drop_last = False
  shuffle = True
  if shuffle:
    sampler = cbs.RandomEpochSampler(dataset, epochs=epochs)
  else:
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)

  batch_sampler = cbs.CycleBatchSampler(sampler, batch_size, drop_last,
                                        schedule=schedule,
                                        long_cycle_bs_scale=long_cycle)
  dataloader = DL.DataLoader(dataset, num_workers=12, batch_sampler=batch_sampler, pin_memory=True)

  return dataloader, dataset, schedule[1:]


def run(init_lr=0.1, max_epochs=100, mode='rgb', root='/data/add_disk0/kumarak/Charades_v1_rgb',
        train_split='./data/charades.json', batch_size=8*2, save_model=''):
    # setup dataset
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':180, 'M':256, 'XL':360}[X3D_VERSION]
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION]
    num_frames = 80

    test_transforms = None

    dataloader, dataset, lr_schedule = setup_data(root, train_split, mode, batch_size, max_epochs, crop_size, resize_size, num_frames)

    val_dataset = Dataset_Full(train_split, 'testing', root, mode, test_transforms, resize_size=resize_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    print('Total steps:', lr_schedule[-1])
    print('datasets created')


    # setup the model
    if mode == 'flow':
        x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=157, n_input_channels=2)
        #x3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
        #save_model = 'models/flow_temp_'
    else:
        x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=157, n_input_channels=3)
        #x3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        save_model = 'models/x3d_multigrid_rgb_'
    #x3d.replace_logits(157)
    #x3d.load_state_dict(torch.load('/ssd/models/000920.pt'))

    #flops, params = profile(x3d, inputs=torch.randn(1,1, 3, 80//gamma_tau, crop_size, crop_size))
    #print(flops, params)

    x3d.cuda()

    #summary(x3d, (3, 80//gamma_tau, crop_size, crop_size))

    x3d = nn.DataParallel(x3d)
    print('model loaded')

    lr = 0.001 #0.00125 * (batch_size/4)
    long_cycle_lr_scale = [8, 0.5, 0.5, 0.5]
    #optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    optimizer = optim.Adam(x3d.parameters(), lr=lr, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)


    num_steps_per_update = 4 #4 * 1 # accum gradient
    steps = 0
    epochs = 0
    last_long = -1
    val_apm = APMeter()
    tr_apm = APMeter()
    # train it
    while epochs < max_epochs:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in 5*['train']+['val']:
            if phase == 'train':
                x3d.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                x3d.train(False)  # Set model to evaluate mode
                torch.autograd.set_grad_enabled(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if phase == 'train':
                    inputs, labels, long_ind = data
                    #print(inputs.shape)
                    long_ind = long_ind[0].item()
                    if long_ind != last_long:
                        last_long = long_ind
                        for g in optimizer.param_groups:
                          g['lr'] *= long_cycle_lr_scale[long_ind]
                          print('LR: ', g['lr'])
                else:
                    inputs, labels = data

                # Gamma_tau sampling ****** MAY NOT BE GOOD FOR LOCALIZATION TASKS ******
                inputs = inputs[:,:,::gamma_tau,:,:]
                labels = labels[:,:,::gamma_tau]

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = x3d(inputs)
                # upsample to input size
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')
                probs = F.sigmoid(per_frame_logits)

                if phase == 'train':
                    for b in range(labels.shape[0]):
                        tr_apm.add(probs[b].transpose(0,1).detach().cpu().numpy(), labels[b].transpose(0,1).cpu().numpy())
                else:
                    for b in range(labels.shape[0]):
                        val_apm.add(probs[b].transpose(0,1).detach().cpu().numpy(), labels[b].transpose(0,1).cpu().numpy())

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item()

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item()

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.item()

                if phase == 'train':
                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0 or steps == 1:
                        tr_map = tr_apm.value().mean()
                        tr_apm.reset()
                        print (' Epoch:{} {} steps: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                            steps, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10, tr_map))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
                    if steps % 100 == 0:
                        # save model
                        torch.save(x3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                val_map = val_apm.value().mean()
                val_apm.reset()
                print (' Epoch:{} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                    tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))



if __name__ == '__main__':
    # need to add argparse
    run()
    #run(mode=args.mode, root=args.root, save_model=args.save_model)
