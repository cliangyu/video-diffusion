import argparse
import os
import pickle
# from https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish  --------------------------------------------------------------
import signal
import time

import cv2
import imageio
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
from invertedai_simulate.interface import IAIEnv, ServerTimeoutError
from invertedai_simulate.utils import (ClientSideBoundingBoxes, Res,
                                       Resolution, SensorSettings)


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# --------------------------------------------------------------

with timeout(seconds=1800):
    IN_COLAB = False

    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir',
                        type=str,
                        help='Where to save the generated videos/coords.')
    parser.add_argument(
        '--port',
        type=int,
        default=5555,
        help=
        '5555 is the server with other cars. Currently there are no other servers',
    )
    parser.add_argument('--max_traffic', type=int, default=200)
    parser.add_argument('--max_pedestrian', type=int, default=200)
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--videos_per_trajectory', type=int, default=5)
    args = parser.parse_args()

    def_res = Resolution(args.res, args.res)
    scale = 1

    sensors_dict = {
        'front-cam': {
            'sensor_type': 'camera',
            'camera_type': 'rgb-camera',
            'bounding_box': False,
            'track_actor_types': SensorSettings.Available_Tracked_Actors,
            'show_bounding_boxes': False,
            'world_sensor': False,
            'resolution': def_res,
            'location': SensorSettings.Location(x=2, z=2, y=0),
            'rotation': SensorSettings.Rotation(yaw=0, roll=0, pitch=0),
            'fov': 120.0,
        },
    }

    fake_parser = argparse.ArgumentParser()
    IAIEnv.add_config(fake_parser)
    config = fake_parser.parse_args(
        ['--client_id', 'mycompany', '--enable_progress_spinner', '0'])
    server_ip = 'simulate.inverted.ai'
    config.zmq_server_address = f'{server_ip}:{args.port}'
    env = IAIEnv(config)
    n_traffic = np.random.randint(args.max_traffic + 1)
    n_pedestrian = np.random.randint(args.max_pedestrian + 1)
    town = np.random.choice(['Town01', 'Town02', 'Town03', 'Town04'])
    # weather = np.random.choice(['ClearNoon', 'SoftRainNoon', 'HardRainNoon','ClearSunset', 'WetCloudySunset', 'SoftRainSunset', 'Random'])  this is not as random as possible
    world_parameters = dict(
        carlatown=town,
        traffic_count=n_traffic,
        pedestrian_count=n_pedestrian,
        weather='Random',
    )

    obs = env.set_scenario('egodriving',
                           world_parameters=world_parameters,
                           sensors=sensors_dict)

    dim = (scale * def_res.width, scale * def_res.height)
    width = np.sum([
        sensors_dict[sns]['resolution'].width * scale for sns in sensors_dict
        if sensors_dict[sns]['sensor_type'] == 'camera'
    ])
    height = np.max([
        sensors_dict[sns]['resolution'].height * scale for sns in sensors_dict
        if sensors_dict[sns]['sensor_type'] == 'camera'
    ])
    full_res = Resolution(width, height)

    action = [0.0, 0.0]
    obs, reward, done, info = env.step(action)


def reset_frames():
    return {'images': [], 'coords': [], 'actions': []}


frames = reset_frames()

video_length = 1000


def get_save_name(index, mode='video', ext='pt'):
    return os.path.join(args.save_dir, f'{mode}_{index}.{ext}')


def next_save_index():
    i = 0
    while os.path.exists(get_save_name(i)):
        i += args.videos_per_trajectory
    return i


trajectory_index = next_save_index()

walltime = time.time()
for i in range(0, video_length * args.videos_per_trajectory):
    try:
        with timeout(seconds=10):
            action = info['expert_action']
            obs, reward, done, info = env.step(action)
            frames['images'].append(obs['sensor_data']['front-cam']['image'])
            frames['coords'].append(obs['compact_vector'])
            frames['actions'].append(action)
    except TimeoutError:
        print('\n\nTimed out!!!!!! Exiting.\n\n')
        exit()
    if (i + 1) % video_length == 0:
        save_index = trajectory_index + i // video_length
        # save
        video = torch.stack([torch.tensor(img) for img in frames['images']])
        torch.save(video, get_save_name(save_index, 'video', 'pt'))
        imageio.mimwrite(
            get_save_name(save_index, 'video', 'mp4'),
            frames['images'],
            fps=10,
            quality=7,
        )
        coords = np.array(frames['coords'])
        np.save(get_save_name(save_index, 'coords', 'npy'), coords)
        actions = np.array(frames['actions'])
        np.save(get_save_name(save_index, 'actions', 'npy'), coords)
        pickle.dump(world_parameters,
                    open(get_save_name(save_index, 'config', 'pkl'), 'wb'))
        frames = reset_frames()
        print(
            f'generated {video_length} frames in {time.time()-walltime} seconds'
        )
        walltime = time.time()

with timeout(600):
    print(env.end_simulation())
    env.close()
