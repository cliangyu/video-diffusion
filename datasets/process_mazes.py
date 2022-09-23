# loosely based on https://github.com/iShohei220/torch-gqn/blob/c0156c72f4e63ca6523ab8d9a6f6b3ce9e0e391d/dataset/convert2torch.py
#
# download mazes before running this using instructions at https://github.com/deepmind/gqn-datasets

import collections
import gzip
import io
import os
import time
from multiprocessing import Process

import tensorflow as tf
import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size'],
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def preprocess_frames(dataset_info, example, jpeg='False'):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    if not jpeg:
        frames = tf.map_fn(
            _convert_frame_data,
            tf.reshape(frames, [-1]),
            dtype=tf.float32,
            back_prop=False,
        )
        dataset_image_dimensions = tuple([dataset_info.frame_size] * 2 + [3])
        frames = tf.reshape(frames, (-1, dataset_info.sequence_size) +
                            dataset_image_dimensions)
        if 64 and 64 != dataset_info.frame_size:
            frames = tf.reshape(frames, (-1, ) + dataset_image_dimensions)
            new_frame_dimensions = (64, ) * 2 + (3, )
            frames = tf.image.resize_bilinear(frames,
                                              new_frame_dimensions[:2],
                                              align_corners=True)
            frames = tf.reshape(frames, (-1, dataset_info.sequence_size) +
                                new_frame_dimensions)
    return frames


def preprocess_cameras(dataset_info, example, raw):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(raw_pose_params,
                                 [-1, dataset_info.sequence_size, 5])
    if not raw:
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw),
             tf.cos(yaw),
             tf.sin(pitch),
             tf.cos(pitch)],
            axis=2)
        return cameras
    else:
        return raw_pose_params


def _get_dataset_files(dataset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)
    if mode == 'train':
        num_files = dataset_info.train_size
    else:
        num_files = dataset_info.test_size

    files = sorted(os.listdir(base))

    return [os.path.join(base, file) for file in files]


def encapsulate(frames, cameras):
    return Scene(cameras=cameras, frames=frames)


def convert_raw_to_numpy(dataset_info, raw_data, path, jpeg=False):
    feature_map = {
        'frames':
        tf.FixedLenFeature(shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras':
        tf.FixedLenFeature(shape=[dataset_info.sequence_size * 5],
                           dtype=tf.float32),
    }
    example = tf.parse_single_example(raw_data, feature_map)
    frames = preprocess_frames(dataset_info, example, jpeg)
    cameras = preprocess_cameras(dataset_info, example, jpeg)
    with tf.train.SingularMonitoredSession() as sess:
        frames = sess.run(frames)
        cameras = sess.run(cameras)
    torch.save(frames, path)


def show_frame(frames, scene, views):
    import matplotlib

    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt

    plt.imshow(frames[scene, views])
    plt.show()


if __name__ == '__main__':
    orig_dataset = 'mazes'
    torch_dataset_path = f'{orig_dataset}-torch'
    try:
        os.mkdir(torch_dataset_path)
    except FileExistsError:
        pass

    dataset_info = DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300,
    )

    for split in ['train', 'test']:
        torch_split_path = os.path.join(torch_dataset_path, split)
        try:
            os.mkdir(torch_split_path)
        except FileExistsError:
            pass
        file_names = _get_dataset_files(dataset_info, split, '.')
        tot = 0
        for file in file_names:
            engine = tf.python_io.tf_record_iterator(file)
            for i, raw_data in enumerate(engine):
                path = os.path.join(torch_split_path, f'{tot+i}.pt')
                if not os.path.exists(path):
                    print(f' [-] converting scene {file}-{i} into {path}')
                    p = Process(
                        target=convert_raw_to_numpy,
                        args=(dataset_info, raw_data, path, True),
                    )
                    p.start()
                    p.join()  # surely this means we are not using parallelism?
                else:
                    print(path, 'exists')
            tot += i

        print(f' [-] {tot} scenes in the {split} dataset')
