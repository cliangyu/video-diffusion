from configparser import MAX_INTERPOLATION_DEPTH
from PIL import Image
import blobfile as bf
import os
import io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import ToTensor, Resize

NO_MPI = ('NO_MPI' in os.environ)
if not NO_MPI:
    from mpi4py import MPI

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    # Disable all GPUS (this avoids tensorflow allocating the whole GPU, causing problems for pytorch)
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except ModuleNotFoundError:
    print('WARNING: Failed tensorflow import.')


video_data_paths_dict = {
    "minerl":  "datasets/minerl_navigate",
    "mazes":  "datasets/mazes-torch",
    "bouncy_balls":  "datasets/bouncing_balls_100",
}

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

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
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=0 if NO_MPI else MPI.COMM_WORLD.Get_rank(),
        num_shards=1 if NO_MPI else MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_video_data(dataset_name, batch_size, T, deterministic=False, train=True):
    # NOTE this is just for loading training data (not test)
    data_path = video_data_paths_dict[dataset_name]
    if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
        data_path = os.path.join(os.environ["DATA_ROOT"], data_path)
    shard = 0 if NO_MPI else MPI.COMM_WORLD.Get_rank()
    num_shards = 1 if NO_MPI else MPI.COMM_WORLD.Get_size()
    def get_loader(dataset):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=(not deterministic), num_workers=1, drop_last=True
        )
    if dataset_name == "minerl":
        loader = MineRLDataLoader(
            data_path, batch_size,
            seq_len=T,
            drop_last=True,
            deterministic=deterministic,
            shard=shard,
            num_shards=num_shards,)
    elif dataset_name == "mazes":
        data_path = os.path.join(data_path, "train")
        dataset = MazesDataset(data_path, shard=shard, num_shards=num_shards, T=T, train=train)
        loader = get_loader(dataset)
    elif dataset_name == "bouncy_balls":
        data_path = os.path.join(data_path, "train.pt")
        dataset = TensorVideoDataset(data_path, shard=shard, num_shards=num_shards)
        loader = get_loader(dataset)
    else:
        raise Exception("no dataset", dataset_name)
    if train:
        while True:
            yield from loader


def get_test_dataset(data_path, T):
    if "minerl" in data_path:
        def _process_seq(seq):
            seq = tf.expand_dims(seq, 0)
            seq = tf.cast(seq, tf.float32) / 255.0
            seq = seq * 2 - 1
            seq = tf.transpose(seq, [0, 1, 4, 2, 3])
            return seq
        dataset = tfds.load('minerl_navigate', shuffle_files=False, data_dir=os.path.dirname(data_path))["test"]
        dataset = dataset.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(_process_seq(x))
        )
        numpy_dataset = np.stack([x.numpy()[:T] for x in dataset])
        dataset = TensorDataset(torch.as_tensor(numpy_dataset), torch.zeros(len(numpy_dataset)))
    return dataset


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class TensorVideoDataset(Dataset):

    def __init__(self, tensor_path, shard=0, num_shards=1):
        super().__init__()
        tensor = torch.load(tensor_path)
        self.local_tensor = self.preprocess(tensor[shard:][::num_shards])
        self.grayscale = (self.local_tensor.shape[2] == 1)

    def preprocess(self, tensor):
        # renormalise from [0, 1] top [-1, 1]
        return 2*tensor - 1

    def __len__(self):
        return len(self.local_tensor)

    def __getitem__(self, idx):
        vid = self.local_tensor[idx]
        if self.grayscale:
            vid = vid.expand(-1, 3, -1, -1)  # network is designed for RGB
        return vid, {}


class MineRLDataLoader:
    def __init__(self, path, batch_size, seq_len,
                 shard=0, num_shards=1, train=True,
                 drop_last=True, deterministic=False):

        assert seq_len is not None
        self._seq_len = seq_len
        self._data_seq_len = 500

        assert shard == 0, "Distributed training is not supported by the MineRL dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the MineRL dataset yet."
        
        # Most of this initialization is taken from https://github.com/vaibhavsaxena11/cwvae/blob/master/data_loader.py
        if train:
            dataset = tfds.load('minerl_navigate', shuffle_files=not deterministic, data_dir=os.path.dirname(path))["train"]
        else:
            dataset = tfds.load('minerl_navigate', shuffle_files=not deterministic, data_dir=os.path.dirname(path))["test"]

        dataset = dataset.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(self._process_seq(x))
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_last,
                                num_parallel_calls=tf.data.AUTOTUNE,
                                deterministic=deterministic)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if train and not deterministic:
            dataset = dataset.shuffle(10 * batch_size)
        self.dataset = dataset

    def _process_seq(self, seq):
        seq = tf.expand_dims(seq, 0)
        seq = tf.cast(seq, tf.float32) / 255.0
        seq = seq * 2 - 1
        seq = tf.transpose(seq, [0, 1, 4, 2, 3])
        return seq

    def __iter__(self):
        for batch in self.dataset:
            batch = batch.numpy()
            video_length = batch.shape[1]
            start_idx = np.random.randint(low=0, high=video_length - self._seq_len + 1)
            yield torch.as_tensor(batch[:, start_idx:start_idx+self._seq_len]), {}


class MazesDataset:
    """ from https://github.com/iShohei220/torch-gqn/blob/master/gqn_dataset.py .
    """
    def __init__(self, path, shard, num_shards, T):
        assert shard == 0, "Distributed training is not supported by the MineRL dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the MineRL dataset yet."
        self.path = path
        self.T = T

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        path = os.path.join(self.path, "{}.pt".format(idx))
        data = torch.load(path)
        if self.T < len(data):
            start_i = np.random.randint(len(data) - self.T + 1)
            data = data[start_i:start_i+self.T]
        # resizes from 84x84 to 64x64
        byte_to_tensor = lambda x: ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))
        video = torch.stack([byte_to_tensor(frame) for frame in data])
        video = 2*video - 1
        return video, {}
