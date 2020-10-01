import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np
import pkbar
from apmeter import APMeter

import x3d as resnet_x3d

from kinetics_multigrid import Kinetics
from kinetics import Kinetics as Kinetics_val

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import cycle_batch_sampler as cbs
import dataloader as DL

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


KINETICS_TRAIN_ROOT = '/nfs/bigneuron/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/train_frames'
KINETICS_TRAIN_ANNO = '/nfs/bigneuron/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/kinetics400/train.json'
KINETICS_VAL_ROOT = '/nfs/bigneuron/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/valid_frames'
KINETICS_VAL_ANNO = '/nfs/bigneuron/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/kinetics400/validate.json'
KINETICS_CLASS_LABELS = '/nfs/bigneuron/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/kinetics400/labels.txt'
KINETICS_MEAN = [110.63666788/255, 103.16065604/255, 96.29023126/255]
KINETICS_STD = [38.7568578/255, 37.88248729/255, 40.02898126/255]
KINETICS_DATASET_SIZE = {'train':220000, 'val':17500}

BS = 8
BS_UPSCALE = 16 # CHANGE WITH GPU AVAILABILITY
INIT_LR = (1.6/1024)*(BS*BS_UPSCALE)
SCHEDULE_SCALE = 4
EPOCHS = (60000 * 1024 * 1.5)/220000 #(~420)

LONG_CYCLE = [8, 4, 2, 1]
LONG_CYCLE_LR_SCALE = [8, 0.5, 0.5, 0.5]
GPUS = 4
BASE_BS_PER_GPU = BS * BS_UPSCALE // GPUS # FOR SPLIT BN
CONST_BN_SIZE = 8

X3D_VERSION = 'M' # ['S', 'M', 'XL']


def setup_data(batch_size, num_steps_per_update, epochs, iterations_per_epoch, cur_iterations, crop_size, resize_size, num_frames, gamma_tau):

  num_iterations = int(epochs * iterations_per_epoch)
  schedule = [int(i*num_iterations) for i in [0, 0.4, 0.65, 0.85, 1]]

  train_transforms = {
      'spatial':  Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                           RandomHorizontalFlip(),
                           ToTensor(255),
                           Normalize(KINETICS_MEAN, KINETICS_STD)]),
      'temporal': TemporalRandomCrop(num_frames, gamma_tau),
      'target':   ClassLabel()
  }

  dataset = Kinetics(
          KINETICS_TRAIN_ROOT,
          KINETICS_TRAIN_ANNO,
          KINETICS_CLASS_LABELS,
          'train',
          spatial_transform=train_transforms['spatial'],
          temporal_transform=train_transforms['temporal'],
          target_transform=train_transforms['target'],
          sample_duration=num_frames)

  drop_last = False
  shuffle = True
  if shuffle:
    sampler = cbs.RandomEpochSampler(dataset, epochs=epochs)
  else:
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)

  batch_sampler = cbs.CycleBatchSampler(sampler, batch_size, drop_last,
                                        schedule=schedule,
                                        cur_iterations = cur_iterations,
                                        long_cycle_bs_scale=LONG_CYCLE)
  dataloader = DL.DataLoader(dataset, num_workers=12, batch_sampler=batch_sampler, pin_memory=True)

  schedule[-2] = (schedule[-2]+schedule[-1])//2 # FINE TUNE LAST PHASE, HALF WITH PREV_LR AND HALF WITH REDUCED_LR

  return dataloader, dataset, schedule[1:]



# max_epochs = int(EPOCHS/SCHEDULE_SCALE)
def run(init_lr=INIT_LR, warmup_steps=8000, max_epochs=120, batch_size=BS*BS_UPSCALE):

    frames=80
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5*2, 'XL':5}[X3D_VERSION] # 'M':5 FOR LONGER SCHEDULE, NUM OF GPUS INCREASE

    st_steps = 204000 #0 # FOR LR WARM-UP
    load_steps = 204000 #0 # FOR LOADING AND PRINT SCHEDULE
    steps = 204000 #0
    epochs = 118 #0
    num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = KINETICS_DATASET_SIZE['train']//batch_size
    val_iterations_per_epoch = KINETICS_DATASET_SIZE['val']//batch_size
    max_steps = iterations_per_epoch * max_epochs

    last_long = -2

    dataloader, dataset, lr_schedule = setup_data(batch_size, num_steps_per_update, max_epochs, iterations_per_epoch,
                                                cur_iterations, crop_size, resize_size, frames, gamma_tau)

    lr_schedule = [i//num_steps_per_update for i in lr_schedule]

    validation_transforms = {
        'spatial':  Compose([CenterCropScaled(crop_size), #CenterCrop(crop_size),
                             ToTensor(255),
                             Normalize(KINETICS_MEAN, KINETICS_STD)]),
        'temporal': TemporalRandomCrop(frames, gamma_tau),
        'target':   ClassLabel()
    }

    val_dataset = Kinetics_val(
            KINETICS_VAL_ROOT,
            KINETICS_VAL_ANNO,
            KINETICS_CLASS_LABELS,
            'validate',
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=frames,
            gamma_tau=gamma_tau,
            crops=3)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', lr_schedule[-1]*num_steps_per_update, 'Total steps:', lr_schedule[-1])
    print('datasets created')


    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3,
                                    dropout=0.5, base_bn_splits=BASE_BS_PER_GPU//CONST_BN_SIZE)
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt') # SET steps=0, OR REMOVE optimizer,lr_sched LOADING
    #x3d.load_state_dict(load_ckpt['model_state_dict'])
    save_model = 'models/x3d_multigrid_kinetics_rgb_sgd_'

    RESTART = False
    if steps>0:
        load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_'+str(load_steps).zfill(6)+'.pt')
        cur_long_ind = load_ckpt['long_ind']
        bn_splits = x3d.update_bn_splits_long_cycle(LONG_CYCLE[cur_long_ind])
        x3d.load_state_dict(load_ckpt['model_state_dict'])
        last_long = cur_long_ind
        RESTART = True

    x3d.cuda()
    #summary(x3d, (3, frames//gamma_tau, crop_size, crop_size))
    x3d = nn.DataParallel(x3d)
    print('model loaded')

    lr = init_lr
    print ('INIT LR: %f'%lr)

    optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)
    if steps>0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion = nn.CrossEntropyLoss()

    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in 4*['train']+['val']: #['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)
            if phase == 'train':
                x3d.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                x3d.train(False)  # Set model to evaluate mode
                _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS
                torch.autograd.set_grad_enabled(False)

            tot_loss = 0.0
            tot_cls_loss = 0.0
            tot_acc = 0.0
            tot_corr = 0.0
            tot_dat = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for i,data in enumerate(dataloaders[phase]):
                num_iter += 1
                bar.update(i)
                if phase == 'train':
                    if i> iterations_per_epoch:
                        break
                    inputs, labels, long_ind, stats = data

                    long_ind = long_ind[0].item()
                    if long_ind != last_long:
                        bn_splits = x3d.module.update_bn_splits_long_cycle(LONG_CYCLE[long_ind]) # UPDATE BN SPLITS FOR LONG CYCLES
                        lr_scale_fact = LONG_CYCLE[long_ind] if (last_long==-2 or long_ind==-1) else LONG_CYCLE_LR_SCALE[long_ind] # WHEN RESTARTING TRAINING AT DIFFERENT LONG CYCLES / AT LAST CYCLE
                        last_long = long_ind
                        for g in optimizer.param_groups:
                          g['lr'] *= lr_scale_fact
                          lr = g['lr']
                        print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, lr)
                    elif RESTART:
                        RESTART = False
                        print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, optimizer.state_dict()['param_groups'][0]['lr'])

                else:
                    inputs, labels = data
                    b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
                    inputs = inputs.view(b*n,c,t,h,w)

                inputs = inputs.cuda() # B 3 T W H
                labels = labels.unsqueeze(1).cuda() # B 1

                logits = x3d(inputs) # B C 1

                if phase == 'train':
                    #logits_sm = F.softmax(logits, dim=1) # not necessary
                    _, preds = torch.max(logits, 1)
                else:
                    logits = logits.view(b,n,logits.shape[1],1) # FOR MULTIPLE TEMPORAL CROPS
                    logits_sm = F.softmax(logits, dim=2)
                    logits_sm = torch.mean(logits_sm, 1)
                    logits = torch.mean(logits, 1)
                    _, preds = torch.max(logits_sm, 1)

                cls_loss = criterion(logits, labels)
                tot_cls_loss += cls_loss.item()

                # Calculate top-1 accuracy
                correct = torch.sum(preds == labels.data)
                tot_corr += correct.double()
                tot_dat += logits.shape[0]

                loss = cls_loss/num_steps_per_update
                tot_loss += loss.item()

                if phase == 'train':
                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    lr_warmup(lr, steps-st_steps, warmup_steps, optimizer) # USE ONLY AT THE START, AVOID OVERLAP WITH LONG_CYCLE CHANGES
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    s_times = iterations_per_epoch//2
                    if (steps-load_steps) % s_times == 0:
                        tot_acc = tot_corr/tot_dat
                        print (' Epoch:{} {} steps: {} Cls Loss: {:.4f} Tot Loss: {:.4f} Acc: {:.4f}'.format(epochs, phase,
                            steps, tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tot_acc))
                        tot_loss = tot_cls_loss = tot_acc = tot_corr = tot_dat = 0.
                    if steps % (1000*4) == 0:
                        ckpt = {'model_state_dict': x3d.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_sched.state_dict(),
                                'long_ind': long_ind}
                        torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                tot_acc = tot_corr/tot_dat
                print (' Epoch:{} {} Cls Loss: {:.4f} Tot Loss: {:.4f} Acc: {:.4f}'.format(epochs, phase,
                    tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, tot_acc))
                tot_loss = tot_cls_loss = tot_acc = tot_corr = tot_dat = 0.



def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr


def print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, lr):
    bs = batch_size * LONG_CYCLE[long_ind]
    if long_ind in [0,1]:
        bs = [bs*j for j in [2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{}) W/H ({},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], stats[2][0], stats[3][0], bn_splits, long_ind))
    else:
        bs = [bs*j for j in [4,2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{},{}) W/H ({},{},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], bs[2], stats[1][0], stats[2][0], stats[3][0], bn_splits, long_ind))


if __name__ == '__main__':
    run()
