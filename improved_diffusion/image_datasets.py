import glob
import io
import os
import shutil
from configparser import MAX_INTERPOLATION_DEPTH
from pathlib import Path

import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import Resize, ToTensor

from .test_util import Protect

NO_MPI = 'NO_MPI' in os.environ
if not NO_MPI:
    from mpi4py import MPI

video_data_paths_dict = {
    'minerl':
    'datasets/minerl_navigate-torch',
    'mazes':
    'datasets/mazes-torch',
    'mazes_cwvae':
    'datasets/gqn_mazes-torch',
    'bouncy_balls':
    'datasets/bouncing_balls_100',
    'carla_with_traffic':
    'datasets/carla/with-traffic',
    'carla_no_traffic':
    'datasets/carla/no-traffic',
    'carla_town02_no_traffic':
    'datasets/carla/town02-no-traffic',
    'carla_no_traffic_variable_length':
    'datasets/carla/no-traffic-variable-length',
}

default_T_dict = {
    'minerl': 500,
    'mazes': 300,
    'mazes_cwvae': 300,
    'ucf101': 300,
    'bouncy_balls': 100,
    'carla_with_traffic': 1000,
    'carla_no_traffic': 1000,
    'carla_town02_no_traffic': 1000,
}

default_image_size_dict = {
    'minerl': 64,
    'mazes': 64,
    'mazes_cwvae': 64,
    'ucf101': 64,
    'bouncy_balls': 32,
    'carla_with_traffic': 128,
    'carla_no_traffic': 128,
    'carla_town02_no_traffic': 128,
}

default_iterations_dict = {
    'minerl': 850000,
    'mazes': 950000,
    'mazes_cwvae': 950000,
    'ucf101': 950000,
    'bouncy_balls': 950000,
    'carla_with_traffic': 500000,
    'carla_no_traffic': 500000,
    'carla_town02_no_traffic': 500000,
}


def load_data(*,
              data_dir,
              batch_size,
              image_size,
              class_cond=False,
              deterministic=False):
    """For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError('unspecified data directory')
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split('_')[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=torch.distributed.get_rank()
        if NO_MPI else MPI.COMM_WORLD.Get_rank(),
        num_shards=torch.cuda.device_count()
        if NO_MPI else MPI.COMM_WORLD.Get_size(),
        # shard=0 if NO_MPI else MPI.COMM_WORLD.Get_rank(),
        # num_shards=1 if NO_MPI else MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            drop_last=True)
    else:
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1,
                            drop_last=True)
    while True:
        yield from loader


def load_video_data(
    dataset_name,
    batch_size,
    T=None,
    image_size=None,
    deterministic=False,
    num_workers=1,
    data_path=None,
):
    # NOTE this is just for loading training data (not test)
    if data_path is None:
        data_path = video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    image_size = (default_image_size_dict[dataset_name]
                  if image_size is None else image_size)

    if 'DATA_ROOT' in os.environ and os.environ['DATA_ROOT'] != '':
        data_path = os.path.join(os.environ['DATA_ROOT'], data_path)
    # shard = 0 if NO_MPI else MPI.COMM_WORLD.Get_rank()
    # num_shards = 1 if NO_MPI else MPI.COMM_WORLD.Get_size()
    if dist.is_initialized():
        shard = torch.distributed.get_rank(
        ) if NO_MPI else MPI.COMM_WORLD.Get_rank()
        num_shards = torch.cuda.device_count(
        ) if NO_MPI else MPI.COMM_WORLD.Get_size()
    else:
        shard = 0
        num_shards = 1

    def get_loader(dataset):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=(not deterministic),
                          num_workers=num_workers,
                          drop_last=True,
                          pin_memory=True,
                          persistent_workers=True)

    if dataset_name == 'minerl':
        data_path = os.path.join(data_path, 'train')
        dataset = MineRLDataset(data_path,
                                shard=shard,
                                num_shards=num_shards,
                                image_size=image_size,
                                T=T)
    elif dataset_name == 'mazes':
        raise Exception('Deprecated dataset.')
        data_path = os.path.join(data_path, 'train')
        dataset = MazesDataset(data_path,
                               shard=shard,
                               num_shards=num_shards,
                               T=T)
    elif dataset_name == 'mazes_cwvae':
        data_path = os.path.join(data_path, 'train')
        dataset = GQNMazesDataset(data_path,
                                  shard=shard,
                                  num_shards=num_shards,
                                  image_size=image_size,
                                  T=T)
    elif dataset_name == 'ucf101':
        # dataset = UCF101Dataset(train=True, path=data_path, shard=shard, num_shards=num_shards, T=T)
        config_path = os.path.join(data_path, 'train.json')
        h5path = os.path.join(data_path, 'train.h5')
        dataset = UCF101Dataset(
            config_path=config_path,
            h5path=h5path,
            image_size=image_size,
        )
    elif dataset_name == 'bair_pushing':
        data_path = os.path.join(data_path, 'train')
        dataset = BairPushingDataset(train=True,
                                     path=data_path,
                                     shard=shard,
                                     num_shards=num_shards,
                                     image_size=image_size,
                                     T=T)
    elif dataset_name in [
            'carla_no_traffic',
            'carla_with_traffic',
            'carla_town02_no_traffic',
    ]:
        dataset = CarlaDataset(train=True,
                               path=data_path,
                               shard=shard,
                               num_shards=num_shards,
                               image_size=image_size,
                               T=T)
    elif dataset_name == 'bouncy_balls':
        data_path = os.path.join(data_path, 'train.pt')
        dataset = TensorVideoDataset(
            data_path,
            shard=shard,
            num_shards=num_shards,
            image_size=image_size,
        )
    else:
        raise Exception('no dataset', dataset_name)
    loader = get_loader(dataset)
    while True:
        yield from loader


def get_test_dataset(dataset_name, T=None, image_size=None):
    if dataset_name == 'mazes':
        raise Exception('Deprecated dataset.')
    data_root = Path(os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ
                     and os.environ['DATA_ROOT'] != '' else '.')
    data_path = data_root / video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    image_size = (default_image_size_dict[dataset_name]
                  if image_size is None else image_size)

    if dist.is_initialized():
        shard = torch.distributed.get_rank(
        ) if NO_MPI else MPI.COMM_WORLD.Get_rank()
        num_shards = torch.cuda.device_count(
        ) if NO_MPI else MPI.COMM_WORLD.Get_size()
    else:
        shard = 0
        num_shards = 1

    if dataset_name == 'minerl':
        data_path = os.path.join(data_path, 'test')
        dataset = MineRLDataset(data_path,
                                shard=shard,
                                num_shards=num_shards,
                                image_size=image_size,
                                T=T)
    elif dataset_name == 'mazes':
        data_path = os.path.join(data_path, 'test')
        dataset = MazesDataset(data_path,
                               shard=shard,
                               num_shards=num_shards,
                               image_size=image_size,
                               T=T)
    elif dataset_name == 'mazes_cwvae':
        data_path = os.path.join(data_path, 'test')
        dataset = GQNMazesDataset(data_path,
                                  shard=shard,
                                  num_shards=num_shards,
                                  image_size=image_size,
                                  T=T)
    elif dataset_name in [
            'carla_no_traffic',
            'carla_with_traffic',
            'carla_town02_no_traffic',
    ]:
        dataset = CarlaDataset(train=False,
                               path=data_path,
                               shard=0,
                               num_shards=1,
                               image_size=image_size,
                               T=T)
    else:
        raise Exception('no dataset', dataset_name)
    dataset.set_test()
    return dataset


def get_variable_length_dataset(dataset_name, T):
    assert dataset_name == 'carla_no_traffic'
    return CarlaVariableLengthDataset(T)


def get_train_dataset(dataset_name, T=None, image_size=None):
    if dataset_name == 'mazes':
        raise Exception('Deprecated dataset.')
    data_root = Path(os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ
                     and os.environ['DATA_ROOT'] != '' else '.')
    data_path = data_root / video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    image_size = (default_image_size_dict[dataset_name]
                  if image_size is None else image_size)

    if dataset_name == 'minerl':
        data_path = os.path.join(data_path, 'train')
        dataset = MineRLDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == 'mazes':
        data_path = os.path.join(data_path, 'train')
        dataset = MazesDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name in [
            'carla_no_traffic',
            'carla_with_traffic',
            'carla_town02_no_traffic',
    ]:
        dataset = CarlaDataset(train=True,
                               path=data_path,
                               shard=0,
                               num_shards=1,
                               T=T)
    elif dataset_name == 'mazes_cwvae':
        data_path = os.path.join(data_path, 'train')
        dataset = GQNMazesDataset(data_path, shard=0, num_shards=1, T=T)
    else:
        raise Exception('no dataset', dataset_name)
    return dataset


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split('.')[-1]
        if '.' in entry and ext.lower() in ['jpg', 'jpeg', 'png', 'gif']:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self,
                 resolution,
                 image_paths,
                 classes=None,
                 shard=0,
                 num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[
            shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, 'rb') as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                         resample=Image.BOX)

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(
            round(x * scale) for x in pil_image.size),
                                     resample=Image.BICUBIC)

        arr = np.array(pil_image.convert('RGB'))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y:crop_y + self.resolution,
                  crop_x:crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict['y'] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class TensorVideoDataset(Dataset):
    def __init__(self, tensor_path, shard=0, num_shards=1):
        super().__init__()
        tensor = torch.load(tensor_path)
        self.local_tensor = self.preprocess(tensor[shard:][::num_shards])
        self.grayscale = self.local_tensor.shape[2] == 1

    def preprocess(self, tensor):
        # renormalise from [0, 1] top [-1, 1]
        return 2 * tensor - 1

    def __len__(self):
        return len(self.local_tensor)

    def __getitem__(self, idx):
        vid = self.local_tensor[idx]
        if self.grayscale:
            vid = vid.expand(-1, 3, -1, -1)  # network is designed for RGB
        return vid, {}


class BaseDataset(Dataset):
    """The base class for our video datasets. It is used for datasets where
    each video is stored under <dataset_root_path>/<split> as a single file.
    This class provides the ability of caching the dataset items in a temporary
    directory (if specified as an environment variable DATA_ROOT) as the items
    are read. In other words, every time an item is retrieved from the dataset,
    it will try to load it from the temporary directory first. If it is not
    found, it will be first copied from the original location.

        This class provides a default implementation for __len__ as the number of file in the dataset's original directory.
        It also provides the following two helper functions:
        - cache_file: Given a path to a dataset file, makes sure the file is copied to the temporary directory.
        - get_video_subsequence: Takes a video and a video length as input. If the video length is smaller than the
          input video's length, it returns a random subsequence of the video. Otherwise, it returns the whole video.
        A child class should implement the following methods:
        - getitem_path: Given an index, returns the path to the video file.
        - loaditem: Given a path to a video file, loads and returns the video.
        - postprocess_video: Given a video, performs any postprocessing on the video.

    Args:
        path (str): path to the dataset split
    """
    def __init__(self, path, T):
        super().__init__()
        self.T = T
        self.path = Path(path)
        self.is_test = False

    def __len__(self):
        path = self.get_src_path(self.path)
        return len(list(path.iterdir()))

    def __getitem__(self, idx):
        path = self.getitem_path(idx)
        self.cache_file(path)
        try:
            video = self.loaditem(path)
        except Exception as e:
            print(f'Failed on loading {path}')
            raise e
        video = self.postprocess_video(video)
        return self.get_video_subsequence(video, self.T), {}

    def getitem_path(self, idx):
        raise NotImplementedError

    def loaditem(self, path):
        raise NotImplementedError

    def postprocess_video(self, video):
        raise NotImplementedError

    def cache_file(self, path):
        # Given a path to a dataset item, makes sure that the item is cached in the temporary directory.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            src_path = self.get_src_path(path)
            with Protect(path):
                shutil.copyfile(str(src_path), str(path))

    @staticmethod
    def get_src_path(path):
        """Returns the source path to a file.

        This function is mainly used to cope with my way of handling
        SLURM_TMPDIR on CC. If DATA_ROOT is defined as an environment variable,
        the datasets should be copied under it. This function is called when we
        need the source path from a given path under DATA_ROOT.
        """
        if 'DATA_ROOT' in os.environ and os.environ['DATA_ROOT'] != '':
            # Verify that the path is under
            data_root = Path(os.environ['DATA_ROOT'])
            assert (
                data_root in path.parents
            ), f'Expected dataset item path ({path}) to be located under the data root ({data_root}).'
            src_path = Path(
                *path.parts[len(data_root.parts):]
            )  # drops the data_root part from the path, to get the relative path to the source file.
            return src_path
        return path

    def set_test(self):
        self.is_test = True
        print('setting test mode')

    def get_video_subsequence(self, video, T):
        if T is None:
            return video
        if T < len(video):
            # Take a subsequence of the video.
            start_i = 0 if self.is_test else np.random.randint(
                len(video) - T + 1)
            video = video[start_i:start_i + T]
        assert len(video) == T
        return video


class MazesDataset(BaseDataset):
    """from https://github.com/iShohei220/torch-gqn/blob/master/gqn_dataset.py
    ."""
    def __init__(self, path, shard, num_shards, T):
        # assert (
        #     shard == 0
        # ), 'Distributed training is not supported by the MineRL dataset yet.'
        # assert (
        #     num_shards == 1
        # ), 'Distributed training is not supported by the MineRL dataset yet.'
        super().__init__(path=path, T=T)

    def getitem_path(self, idx):
        return self.path / f'{idx}.pt'

    def loaditem(self, path):
        return torch.load(path)

    def postprocess_video(self, video):
        # resizes from 84x84 to 64x64
        def byte_to_tensor(x):
            return ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))

        video = torch.stack([byte_to_tensor(frame) for frame in video])
        video = 2 * video - 1
        return video


import h5py
import pandas as pd
from torch.utils import data


class UCF101Dataset(data.Dataset):
    def __init__(self, h5path, config_path, img_size=64):
        self.h5file = h5py.File(h5path, 'r')
        self.dset = self.h5file['image']
        self.conf = pd.read_json(config_path)
        self.ind = self.conf.index.tolist()
        self.n_frames = 16
        self.img_size = img_size

    def __len__(self):
        return len(self.conf)

    def _crop_center(self, x):
        if self.img_size == 64:
            x = x[:, :, :, 10:10 + self.img_size]
        elif self.img_size == 192:
            x = x[:, :, :, 32:32 + self.img_size]
        assert x.shape[2] == self.img_size
        assert x.shape[3] == self.img_size
        return x

    def __getitem__(self, i):
        mov_info = self.conf.loc[self.ind[i]]
        length = mov_info.end - mov_info.start
        offset = np.random.randint(
            length - self.n_frames) if length > self.n_frames else 0
        x = self.dset[mov_info.start + offset:mov_info.start + offset +
                      self.n_frames]
        x = self._crop_center(x)
        return torch.tensor((x - 128.0) / 128.0, dtype=torch.float)


class BairPushingDataset(MazesDataset):
    def __init__(self, train, shard, num_shards, *args, **kwargs):
        super().__init__(shard=0, num_shards=1, *args,
                         **kwargs)  # dumy values of
        self.split_path = self.path / f"video_{'train' if train else 'test'}.csv"
        # self.cache_file(self.split_path)
        self.fnames = [
            line.rstrip('\n').split('/')[-1]
            for line in open(self.split_path, 'r').readlines() if '.pt' in line
        ]
        self.fnames = self.fnames[shard::num_shards]
        print(f'Training on {len(self.fnames)} files (Carla dataset).')

        self.videos = []
        for idx in range(len(self.fnames)):
            path = self.getitem_path(idx)
            self.cache_file(path)
            try:
                video = self.loaditem(path)
            except Exception as e:
                print(f'Failed on loading {path}')
                raise e
            video = self.postprocess_video(video)
            self.videos.append(video)

    def __getitem__(self, idx):
        video = self.videos[idx]
        return self.get_video_subsequence(video, self.T), {}

    def getitem_path(self, idx):
        return self.path / self.fnames[idx]

    def postprocess_video(self, video):
        return -1 + 2 * (video.permute(0, 3, 1, 2).float() / 255)

    def __len__(self):
        return len(self.fnames)


class CarlaDataset(MazesDataset):
    def __init__(self, train, shard, num_shards, image_size, *args, **kwargs):
        super().__init__(shard=0, num_shards=1, *args,
                         **kwargs)  # dumy values of
        self.image_size = image_size
        self.split_path = self.path / f"video_{'train' if train else 'test'}.csv"
        # self.cache_file(self.split_path)
        self.fnames = [
            line.rstrip('\n').split('/')[-1]
            for line in open(self.split_path, 'r').readlines() if '.pt' in line
        ]
        self.fnames = self.fnames[shard::num_shards]
        print(f'Training on {len(self.fnames)} files (Carla dataset).')

        self.videos = []
        for idx in range(len(self.fnames)):
            path = self.getitem_path(idx)
            self.cache_file(path)
            try:
                video = self.loaditem(path)
            except Exception as e:
                print(f'Failed on loading {path}')
                raise e
            video = self.postprocess_video(video)
            self.videos.append(video)

    def __getitem__(self, idx):
        video = self.videos[idx]
        return self.get_video_subsequence(video, self.T), {}

    def getitem_path(self, idx):
        return self.path / self.fnames[idx]

    def postprocess_video(self, video):
        video = -1 + 2 * (video.permute(0, 3, 1, 2).float() / 255)
        video = Resize(self.image_size)(video)
        return video

    def __len__(self):
        return len(self.fnames)


class CarlaVariableLengthDataset(CarlaDataset):
    def __init__(self, T):
        path = os.path.join('datasets', 'carla', 'no-traffic-variable-length')
        print('in variable length dataset __init__')
        self.T = T
        print(self.T)
        self.fnames = sorted([
            Path(p).name for p in glob.glob(os.path.join(path, 'video_*.pt'))
        ])
        print(os.path.join(path, 'video_*.pt'))
        print(self.fnames)
        self.path = Path(path)
        print(self.path)
        self.is_test = False


class GQNMazesDataset(BaseDataset):
    """based on https://github.com/iShohei220/torch-
    gqn/blob/master/gqn_dataset.py ."""
    def __init__(self, path, shard, num_shards, T):
        # assert (
        #     shard == 0
        # ), 'Distributed training is not supported by the MineRL dataset yet.'
        # assert (
        #     num_shards == 1
        # ), 'Distributed training is not supported by the MineRL dataset yet.'
        super().__init__(path=path, T=T)

    def getitem_path(self, idx):
        return self.path / f'{idx}.npy'

    def loaditem(self, path):
        return np.load(path)

    def postprocess_video(self, video):
        def byte_to_tensor(x):
            return ToTensor()(x)

        video = torch.stack([byte_to_tensor(frame) for frame in video])
        video = 2 * video - 1
        return video


class MineRLDataset(BaseDataset):
    def __init__(self, path, shard, num_shards, image_size, T):
        # assert (
        #     shard == 0
        # ), 'Distributed training is not supported by the MineRL dataset yet.'
        # assert (
        #     num_shards == 1
        # ), 'Distributed training is not supported by the MineRL dataset yet.'
        super().__init__(path=path, T=T)
        self.image_size = image_size

    def getitem_path(self, idx):
        return self.path / f'{idx}.npy'

    def loaditem(self, path):
        return np.load(path)

    def postprocess_video(self, video):
        def byte_to_tensor(x):
            return ToTensor()(x)

        video = torch.stack([byte_to_tensor(frame) for frame in video])
        video = 2 * video - 1
        video = Resize(self.image_size)(video)
        return video
