import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cycle_batch_sampler as cbs

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    #return torch.from_numpy(pic.transpose([3,0,1,2]))
    return torch.from_numpy(pic)


def load_rgb_frames(image_dir, vid, start, num, t_stride, resize_size, width, height, crop_size, center_crop):
  frames = []
  for i in range(start, start+num, t_stride):
    img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    h,w,c = img.shape
    if w < resize_size or h < resize_size:
        d = float(resize_size)-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)

  frames = np.asarray(frames, dtype=np.float32)
  #print(frames.shape)
  _,h,w,_ = frames.shape
  if center_crop:
    i = int(round((h-crop_size)/2.))
    j = int(round((w-crop_size)/2.))
    frames = frames[:, i:-i, j:-j, :]
  else:
    th = crop_size
    tw = crop_size
    i = random.randint(0, h - th) if h!=th else 0
    j = random.randint(0, w - tw) if w!=tw else 0
    frames = frames[:, i:i+th, j:j+tw, :]

  frames = torch.from_numpy(frames.transpose([3,0,1,2])) # C T H W
  frames = F.interpolate(frames, size=(height, width), mode='bilinear').numpy()
  #print('!', frames.shape)

  return frames


def load_flow_frames(image_dir, vid, start, num, t_stride, resize_size, width, height, crop_size, center_crop):
  frames = []
  for i in range(start, start+num, t_stride):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)

    w,h = imgx.shape
    if w < resize_size or h < resize_size:
        d = float(resize_size)-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)

  frames = np.asarray(frames, dtype=np.float32)
  _,h,w,_ = frames.shape
  if center_crop:
    i = int(round((h-crop_size)/2.))
    j = int(round((w-crop_size)/2.))
    frames = frames[:, i:-i, j:-j, :]
  else:
    th = crop_size
    tw = crop_size
    i = random.randint(0, h - th) if h!=th else 0
    j = random.randint(0, w - tw) if w!=tw else 0
    frames = frames[:, i:i+th, j:j+tw, :]

  frames = torch.from_numpy(frames.transpose([3,0,1,2])) # C T H W
  frames = F.interpolate(frames, size=(height, width), mode='bilinear').numpy()

  return frames


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    pre_data_file = split_file[:-5]+'_'+split+'labeldata.npy'
    if os.path.exists(pre_data_file):
        print('{} exists'.format(pre_data_file))
        dataset = np.load(pre_data_file, allow_pickle=True)
    else:
        print('{} does not exist'.format(pre_data_file))
        i = 0
        for vid in data.keys():
            if data[vid]['subset'] != split:
                continue

            if not os.path.exists(os.path.join(root, vid)):
                continue
            num_frames = len(os.listdir(os.path.join(root, vid)))
            if mode == 'flow':
                num_frames = num_frames//2

            if num_frames < (80+2):
                continue

            label = np.zeros((num_classes,num_frames), np.float32)

            fps = num_frames/data[vid]['duration']
            for ann in data[vid]['actions']:
                for fr in range(0,num_frames,1):
                    if fr/fps > ann[1] and fr/fps < ann[2]:
                        label[ann[0], fr] = 1 # binary classification
            dataset.append((vid, label, data[vid]['duration'], num_frames))
            i += 1
            print(i, vid)
        np.save(pre_data_file, dataset)

    print('dataset size:%d'%len(dataset))
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None,
                resize_size=256, crop_size=224, center_crop=False, num_f=80):

        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.resize_size = resize_size
        self.mode = mode
        self.root = root

        self.center_crop = center_crop
        self.crop_size = crop_size
        self.num_f = num_f
        self.width = crop_size
        self.height = crop_size
        self.long_cycles = [
          (self.num_f//4, int(self.width/np.sqrt(2)), int(self.height/np.sqrt(2))),
          (self.num_f//2, int(self.width/np.sqrt(2)), int(self.height/np.sqrt(2))),
          (self.num_f//2, self.width, self.height),
          (self.num_f, self.width, self.height)]

    def __getitem__(self, index):

        #print(index)
        iteration = index[0]
        index, long_cycle_state = index[1]

        # get short cycle for iter
        short_cycle_state = iteration % 3

        vid, label, dur, nf = self.data[index]

        num_f, width, height = self.long_cycles[long_cycle_state]
        if short_cycle_state == 1:
          width = int(width/np.sqrt(2))
          height = int(height/np.sqrt(2))
        elif short_cycle_state == 2:
          width = width//2
          height = height//2

        t_stride = random.randint(1,1*(self.num_f/num_f)) # 2 times 80 ??
        load_num_f = num_f * t_stride
        start_f = random.randint(1,nf-(load_num_f+1))

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, load_num_f, t_stride, self.resize_size,
                                    width, height, self.crop_size, self.center_crop)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, load_num_f, t_stride, self.resize_size,
                                    width, height, self.crop_size, self.center_crop)
        label = label[:, start_f:start_f+load_num_f:t_stride]

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        #print(imgs.shape, label.shape)
        return video_to_tensor(imgs), torch.from_numpy(label), long_cycle_state

    def __len__(self):
        return len(self.data)
