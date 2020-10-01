import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    torchvision.set_image_backend('accimage')
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'frame_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    data = open(data).read().splitlines()
    for class_label in data: #['labels']
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data.items():
        this_subset = value['subset']
        if this_subset == subset:

            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            elif subset == 'train':
                st = int(value['annotations']['segment'][0])
                end = int(value['annotations']['segment'][1])
                label = value['annotations']['label'].replace(' ','_')
                video_names.append('{}/{}_{}_{}'.format(label, key, str(st).zfill(6), str(end).zfill(6)))
                annotations.append(value['annotations'])
            else:
                label = value['annotations']['label'].replace(' ','_')
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, class_labels, subset, n_samples_for_each_video, sample_duration):

    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(class_labels)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    pre_saved_dataset = os.path.join(root_path, 'labeldata_80.npy')
    if os.path.exists(pre_saved_dataset):
        print('{} exists'.format(pre_saved_dataset))
        dataset = np.load(pre_saved_dataset, allow_pickle=True)
    else:
        dataset = []
        na = 0
        for i in range(len(video_names)):
            if i % 1000 == 0:
                print('dataset loading [{}/{}] N/A {}'.format(i, len(video_names), na))

            video_path = os.path.join(root_path, video_names[i])
            if not os.path.exists(video_path):
                na += 1
                continue

            n_frames = len(os.listdir(video_path))
            if n_frames <= 80+1:
                na += 1
                continue

            begin_t = 1
            end_t = n_frames
            sample = {
                'video': video_path,
                'segment': [begin_t, end_t],
                'n_frames': n_frames,
                'video_id': video_names[i].split('/')[1]
            }
            if len(annotations) != 0:
                sample['label'] = class_to_idx[annotations[i]['label']]
            else:
                sample['label'] = -1

            if n_samples_for_each_video == 1:
                sample['frame_indices'] = list(range(1, n_frames + 1))
                dataset.append(sample)
            else:
                if n_samples_for_each_video > 1:
                    step = max(1,
                               math.ceil((n_frames - 1 - sample_duration) /
                                         (n_samples_for_each_video - 1)))
                else:
                    step = sample_duration
                for j in range(1, n_frames, step):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(
                        range(j, min(n_frames + 1, j + sample_duration)))
                    dataset.append(sample_j)
        np.save(pre_saved_dataset, dataset)

    return dataset, idx_to_class


class Kinetics(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 class_labels,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 gamma_tau=5,
                 crops=10,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, class_labels, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.gamma_tau = gamma_tau
        self.crops = crops
        self.sample_duration = sample_duration
        self.frames = sample_duration//gamma_tau


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        #if self.temporal_transform is not None:
        #    frame_indices = self.temporal_transform(frame_indices)

        # FOR MULTI-CROP TESTING
        frame_indices = frame_indices[::self.gamma_tau]
        step = int((len(frame_indices) - 1 - self.frames)//(self.crops-1))

        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3) # T C H W --> C T H W

        if step == 0:
            clips = [clip[:,:self.frames,...] for i in range(self.crops)]
            clips = torch.stack(clips, 0)
        else:
            clips = [clip[:,i:i+self.frames,...] for i in range(0, step*self.crops, step)]
            clips = torch.stack(clips, 0)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clips, target

    def __len__(self):
        return len(self.data)
