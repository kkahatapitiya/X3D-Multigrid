import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
#parser.add_argument('-save_model', type=str)
parser.add_argument('-root', default='/data/add_disk0/kumarak/Charades_v1_rgb', type=str)
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
from torchsummary import summary

import numpy as np
from barbar import Bar
import pkbar
from apmeter import APMeter

import x3d_classi as resnet_x3d

from kinetics_multigrid import Kinetics
from kinetics import Kinetics as Kinetics_val

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import cycle_batch_sampler as cbs
import dataloader as DL

import warnings
warnings.filterwarnings("ignore")


def setup_data(batch_size, num_steps_per_update, epochs, cur_iterations, crop_size, resize_size, num_frames, gamma_tau):

  # base BS of 64 -> ~250 iterations per epoch for kin
  steps_per_epoch = 220000//(batch_size*num_steps_per_update)
  num_iterations = int(epochs * steps_per_epoch)
  #schedule = [0, int(0.6*num_iterations), int(0.85*num_iterations), num_iterations]
  schedule = [0, int(0.4*num_iterations), int(0.6*num_iterations), int(0.85*num_iterations), num_iterations]
  long_cycle = [8, 4, 2, 1]
  dat_mean = [110.63666788/255, 103.16065604/255, 96.29023126/255]
  dat_std = [38.7568578/255, 37.88248729/255, 40.02898126/255]

  train_transforms = {
      'spatial':  Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                           RandomHorizontalFlip(),
                           ToTensor(255),
                           Normalize(dat_mean, dat_std)]),
      'temporal': TemporalRandomCrop(num_frames, gamma_tau),
      'target':   ClassLabel()
  }

  dataset = Kinetics(
          '/data/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/train_frames',
          '/data/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/kinetics400/train.json',
          'train',
          spatial_transform=train_transforms['spatial'],
          temporal_transform=train_transforms['temporal'],
          target_transform=train_transforms['target'])
  #print(len(dataset))

  drop_last = False
  shuffle = True
  if shuffle:
    sampler = cbs.RandomEpochSampler(dataset, epochs=epochs)
  else:
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)

  batch_sampler = cbs.CycleBatchSampler(sampler, batch_size, drop_last,
                                        schedule=schedule,
                                        cur_iterations = cur_iterations,
                                        long_cycle_bs_scale=long_cycle)
  dataloader = DL.DataLoader(dataset, num_workers=12, batch_sampler=batch_sampler, pin_memory=True)

  return dataloader, dataset, schedule[1:]


#'/data/add_disk0/kumarak/Charades_v1_rgb'
#0.002
def run(init_lr=0.4, warmup_steps=50, max_epochs=100, mode='rgb', root='/data/add_disk0/kumarak/Charades_v1_rgb', #'/nfs/bigdisk/kumarak/datasets/charades/Charades_v1_rgb',
    train_split='../data/charades.json', batch_size=8*1, frames=80, i3d_in = '/data/add_disk0/kumarak/i3d_in/nodir',
    i3d_out = {'logit': '/data/add_disk0/kumarak/i3d_out_new/logit',
                'mixed_4e': '/data/add_disk0/kumarak/i3d_out_new/mixed_4e'}
    ): #, save_model=''):
    # setup dataset
    X3D_VERSION = 'M'
    dat_mean = [110.63666788/255, 103.16065604/255, 96.29023126/255]
    dat_std = [38.7568578/255, 37.88248729/255, 40.02898126/255]
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,320.], 'XL':[360.,450.]}[X3D_VERSION]
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION]

    steps = 1200
    epochs = 46
    num_steps_per_update = 4 * 2 * 4 # accum gradient
    cur_iterations = steps * num_steps_per_update
    steps_per_epoch = 220000//(batch_size*num_steps_per_update)
    val_steps_per_epoch = 17500//(batch_size)
    max_steps = steps_per_epoch * max_epochs


    dataloader, dataset, lr_schedule = setup_data(batch_size, num_steps_per_update, max_epochs, cur_iterations,
                                                crop_size, resize_size, frames, gamma_tau)

    validation_transforms = {
        'spatial':  Compose([CenterCrop(crop_size),
                             ToTensor(255),
                             Normalize(dat_mean, dat_std)]),
        'temporal': TemporalRandomCrop(frames, gamma_tau),
        'target':   ClassLabel()
    }
    #val_dataset = Dataset_Full(train_split, 'testing', root, mode, test_transforms, i3d_in=i3d_in, i3d_out=i3d_out)
    val_dataset = Kinetics_val(
            '/data/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/valid_frames',
            '/data/add_disk0/kumarak/Kinetics/kinetics-downloader/dataset/kinetics400/validate.json',
            'validate',
            n_samples_for_each_video=1,
            spatial_transform=validation_transforms['spatial'],
            temporal_transform=validation_transforms['temporal'],
            target_transform=validation_transforms['target'],
            sample_duration=frames)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total steps:', lr_schedule[-1])
    print('datasets created')

    # setup the model
    if mode == 'flow':
        x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=2)
        #save_model = 'models/flow_temp_'
    else:
        x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3)
        #x3d.load_state_dict(torch.load('models/x3d_rgb_v2_adam_001600.pt'))
        save_model = 'models/x3d_multigrid_kinetics_rgb_v2_adam_'
    #i3d.replace_logits(157)

    x3d.load_state_dict(torch.load('models/x3d_multigrid_kinetics_rgb_v2_adam_'+str(steps).zfill(6)+'.pt'))
    #i3d.load_state_dict(torch.load('models/rgb_seV3_withSupEv_fromCharades_sgd_042000.pt'))
    #i3d.load_state_dict(torch.load('models/rgb_sev3_adam_EndtoEndMidwith1xconv_001200_030.pt'))
    #i3d.load_state_dict(torch.load('models/rgb_sev3_adam_EndtoEnd2xMidwith1xconv_001200_030.pt'))

    x3d.cuda()

    ''' freeze BN layers '''
    #for module in i3d.modules():
    #    if isinstance(module, nn.BatchNorm3d):
    #        module.train(False)
            #module.track_running_stats = False
            #print(module)

    #i3d2.replace_logits(157)
    #i3d2.load_state_dict(torch.load('models/rgb_seV2_12_i3d2_000400.pt'))
    #i3d2.cuda()

    #i3d.freeze('Mixed_5c')

    #for name, param in i3d.named_parameters():
    #    if 'dis_' in name: # or 'Mixed_4f' in name :
    #        param.requires_grad = False

    #for name, param in i3d.named_parameters():
    #    if param.requires_grad:print('updating: {}'.format(name))
        #else:print('frozen: {}'.format(name))

    #summary(i3d, (1, 3, 64, 224, 224))
    x3d = nn.DataParallel(x3d)
    #i3d2 = nn.DataParallel(i3d2)
    print('model loaded')

    lr = init_lr #* batch_size/len(datasets['train'])
    print ('LR:%f'%lr)

    #rw_para=[]; def_para=[];
    #for name, para in i3d.named_parameters():
    #    if '_rw' in name:
    #        rw_para.append(para)
    #    else:
    #        def_para.append(para)


    long_cycle_lr_scale = [8, 0.5, 0.5, 0.5]
    optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.SGD([{'params': def_para},{'params': rw_para, 'lr': lr*10}], lr=lr, momentum=0.9, weight_decay=1e-7)
    #optimizer = optim.Adam(x3d.parameters(), lr=lr, weight_decay=1e-7)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300,1000]) #300
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [2000,3000,4000,5000], gamma=0.5) #300
    #lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0, last_epoch=-1)#, verbose=True)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)
    #lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, min_lr=1e-3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    #for g in optimizer.param_groups:
        #if '_rw' in name:
        #print(g)
        #g['lr'] *= long_cycle_lr_scale[long_ind]

    #num_steps_per_update = 4 * 2 # accum gradient
    #steps = 0#1200#1000 #1800 #400 #0
    #epochs = 0#30#25 #44 #10 #0
    last_long = -1
    #val_apm = APMeter()
    #tr_apm = APMeter()
    # train it
    while epochs < max_epochs:#for epoch in range(num_epochs):
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']+['val']: #*5+['val']:
            bar_st = steps_per_epoch if phase == 'train' else val_steps_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)
            if phase == 'train':
                x3d.train(True)
                #i3d2.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                x3d.train(False)  # Set model to evaluate mode
                #i3d2.train(False)
                torch.autograd.set_grad_enabled(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            tot_dis_loss = 0.0
            tot_acc = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for i,data in enumerate(dataloaders[phase]):
                #for data in dataloaders[phase]:
                num_iter += 1
                bar.update(i)
                if phase == 'train':
                    if i>steps_per_epoch:
                        break
                    inputs, labels, long_ind, stats = data
                    #print(inputs.shape)
                    long_ind = long_ind[0].item()
                    if long_ind != last_long:
                        last_long = long_ind
                        for g in optimizer.param_groups:
                          g['lr'] *= long_cycle_lr_scale[long_ind]
                          #print('LR: ', g['lr'])
                          bs = batch_size * [8,4,2,1][long_ind]; bs = [bs*j for j in [4,2,1]]
                          print('LR {} Frames {} BS ({},{},{}) W/H ({},{},{})'.format(g['lr'], stats[0][0], bs[0], bs[1], bs[2], stats[1][0], stats[2][0], stats[3][0]))
                else:
                    inputs, labels = data
                #print(num_iter)
                # get the inputs
                #inputs, labels, stats = data #
                #print(meta)
                #print(inputs.shape, labels.shape)
                #print(steps, num_iter, inputs.shape, labels.shape) #(B Ch=3 T=64 H=224 W=224) (B C=157 T)

                # Gamma_tau sampling ****** MAY NOT BE GOOD FOR LOCALIZATION TASKS ******
                #inputs = inputs[:,:,::gamma_tau,:,:]
                #labels = labels[:,:,::gamma_tau]

                # wrap them in Variable
                inputs = inputs.cuda() # B 3 T W H
                t = inputs.size(2)
                labels = labels.unsqueeze(1).cuda() #[:,:50,:] # B 1

                #print(inputs.shape, feat1.shape)
                #print([torch.where(labels[0,:,i]==1)[0].cpu().numpy() for i in range(0,labels.shape[2])])
                #print('l_full',torch.argmax(labels[0], dim=0).detach().cpu().numpy())

                ''' #  *** for segmented val input ***
                if phase == 'val':
                    b,c,t,w,h = inputs.shape
                    inputs = inputs[:,:,:(t//64)*64,:,:].view(b,c,-1,64,w,h).permute(0,2,1,3,4,5).reshape(-1,c,64,w,h)
                    inputs2 = inputs2.repeat(inputs.shape[0],1,1,1,1)
                    meta = torch.from_numpy(np.array([[i*64,t] for i in range(t//64)]))
                    labels = labels[:,:,:(t//64)*64]
                    #print(inputs.shape, inputs2.shape, meta.shape)
                '''

                #''' branch 2 '''
                #baseline_logits, reweights = i3d2([inputs2, meta, None, None])  # seV2
                #baseline_logits, reweights = i3d2([inputs2, meta, None])
                #''' branch 1 '''
                #reweights = i3d.module.get_reweights([inputs, meta])  # B C T//16

                logits = x3d(inputs) # seV2 # B C 1
                #per_frame_logits = i3d([inputs, meta, reweights])

                #''' NOT add baseline '''
                #per_frame_logits += torch.mean(baseline_logits, dim=2, keepdim=True)

                # upsample to input size
                #if phase == 'train':
                #logits = F.upsample(logits, t, mode='linear')
                #else:
                #    per_frame_logits = F.upsample(per_frame_logits, 64, mode='linear') # B C T
                #    per_frame_logits = per_frame_logits.permute(1,0,2).reshape(labels.shape[1],-1).unsqueeze(0)
                    #labels = labels[]

                #probs = F.sigmoid(logits)
                #print(labels.shape, probs.shape)
                #print('l_down',[torch.where(labels[0,:,i]==1)[0].cpu().numpy() for i in range(0,labels.shape[2])])
                #print('l_full',torch.argmax(labels[0], dim=0).detach().cpu().numpy())
                #print('prob',probs[0].detach().cpu().numpy())


                #if phase == 'train':
                #    for b in range(labels.shape[0]):
                #        tr_apm.add(probs[b].transpose(0,1).detach().cpu().numpy(), labels[b].transpose(0,1).cpu().numpy())
                #else:
                #    for b in range(labels.shape[0]):
                #        val_apm.add(probs[b].transpose(0,1).detach().cpu().numpy(), labels[b].transpose(0,1).cpu().numpy())

                # compute localization loss
                #loc_loss = 1 * F.binary_cross_entropy_with_logits(logits, labels)
                #tot_loc_loss += loc_loss.item() #data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                #cls_loss = 1 * criterion(torch.max(logits, dim=2)[0], torch.max(labels, dim=2)[0])
                #tot_cls_loss += cls_loss.item() #data[0]

                #print(per_frame_logits.shape, mask1.shape, feat1.shape)
                #distill_loss = F.mse_loss(feat1024.squeeze(4).squeeze(3) * mask1.unsqueeze(1), feat1.squeeze(4).squeeze(3) * mask1.unsqueeze(1))
                #distill_loss = 1 * F.binary_cross_entropy_with_logits(per_frame_logits * input_mask.unsqueeze(1), F.sigmoid(dis_logits) * mask1.unsqueeze(1))
                #tot_dis_loss += distill_loss.item()

                _, preds = torch.max(logits, 1)
                #print(logits.shape, labels.shape)
                cls_loss = criterion(logits, labels)
                tot_cls_loss += cls_loss.item()

                # Calculate accuracy
                correct = torch.sum(preds == labels.data)
                accuracy = correct.double() / logits.shape[0] #batch_size
                tot_acc += accuracy

                # Calculate elapsed time for this step
                #examples_per_second = 64/float(time.time() - start_time)

                # Back-propagation and optimization step
                #loss.backward()
                #optimizer.step()

                # Save statistics
                #accuracies[step] = accuracy.item()
                #losses[step] = loss.item()

                loss = 1 * (cls_loss)/(1 * num_steps_per_update)
                tot_loss += loss.item() #data[0]

                if phase == 'train':
                    loss.backward()
                    '''
                    for name, para in i3d.named_parameters():
                        if '_rw' in name and para.grad is not None:
                            para.grad *= 5
                    '''
                            #print(name,para.grad.view(-1).detach().cpu().numpy()[:5])
                            #print('name',para.grad)
                        #if 'attnt' in name:
                        #    para.grad *= 5
                        #elif 'attn' in name:
                        #    para.grad *= 3

                if num_iter == num_steps_per_update and phase == 'train':
                    lr_warmup(init_lr, steps, warmup_steps, optimizer) # steps
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 5 == 0:
                        #tr_map = tr_apm.value().mean()
                        #tr_apm.reset()
                        #i3d.module.get_attn_para() #### print mu, sigma
                        print (' Epoch:{} {} steps: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Acc: {:.4f}'.format(epochs, phase,
                            steps, tot_loc_loss/(5*num_steps_per_update), tot_cls_loss/(5*num_steps_per_update), tot_loss/5, tot_acc/(5*num_steps_per_update)))
                        # save model
                        #torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = tot_dis_loss = tot_acc = 0.
                    if steps % 100 == 0:
                        #tr_apm.reset()
                        torch.save(x3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        #torch.save(i3d2.module.state_dict(), save_model2+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                #val_map = val_apm.value().mean()
                #lr_sched.step(tot_loss)
                #val_apm.reset()
                print (' Epoch:{} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Acc: {:.4f}'.format(epochs, phase,
                    tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, tot_acc/num_iter))
                tot_loss = tot_loc_loss = tot_cls_loss = tot_dis_loss = tot_acc = 0.


def init_cropping_scales(num_scales, init_scale, factor):
    # Determine cropping scales
    scales = [init_scale]
    for i in range(1, num_scales):
        scales.append(scales[-1] * factor)
    return scales

def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    if cur_steps < warmup_steps:
            lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
            for pg in opt.param_groups:
                pg['lr'] = lr_scale * init_lr

if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root) #, save_model=args.save_model)
