from invertedai_simulate.utils import Res, SensorSettings, Resolution, ClientSideBoundingBoxes
from invertedai_simulate.interface import IAIEnv, ServerTimeoutError
import numpy as np
import os
import time
import cv2
import imageio_ffmpeg
import imageio
import matplotlib.pyplot as plt
import argparse
import torch
IN_COLAB = False

def_res = Resolution(128, 128)
scale = 1

parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=str, help="Where to save the generated videos/coords.")
parser.add_argument('--port', type=int, default=5555, help="5555 is the server with other cars. Currently there are no other servers")
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--n_indices', type=int, default=5)
args = parser.parse_args()

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
config = fake_parser.parse_args(['--client_id', 'mycompany', '--enable_progress_spinner', '0'])
server_ip = "simulate.inverted.ai"
config.zmq_server_address = f"{server_ip}:{args.port}"
env = IAIEnv(config)
world_parameters = dict(carlatown='Town01')

obs = env.set_scenario('egodriving', world_parameters=world_parameters, sensors=sensors_dict)

dim = (scale*def_res.width, scale*def_res.height)
width = np.sum([sensors_dict[sns]['resolution'].width*scale for sns in sensors_dict if sensors_dict[sns]['sensor_type']=='camera'])
height = np.max([sensors_dict[sns]['resolution'].height*scale for sns in sensors_dict if sensors_dict[sns]['sensor_type']=='camera'])
full_res = Resolution(width, height)

action = [0.0, 0.0]
obs, reward, done, info = env.step(action)

def reset_frames():
    return {'images':[], 'coords': [], 'actions': []}
frames = reset_frames()

T = 1000
walltime = time.time()
for i in range(0, T*args.n_indices):
    action = info['expert_action']
    obs, reward, done, info = env.step(action)
    frames['images'].append(obs['sensor_data']['front-cam']['image'])
    frames['coords'].append(obs['compact_vector'])
    frames['actions'].append(action)

    if i % T == T-1:
        i_save = args.start_index + i//T
        video = torch.stack([torch.tensor(img) for img in frames['images']])
        torch.save(video, os.path.join(args.save_dir, f'video_{i_save}.pt'))
        imageio.mimwrite(os.path.join(args.save_dir, f'video_{i_save}.mp4'), frames['images'], fps=10, quality=7)
        coords = np.array(frames['coords'])
        np.save(os.path.join(args.save_dir, f'coords_{i_save}.npy'), coords)
        actions = np.array(frames['actions'])
        np.save(os.path.join(args.save_dir, f'actions_{i_save}.npy'), coords)
        frames = reset_frames()
        print(f'generated {T} frames in {time.time()-walltime} seconds')
        walltime = time.time()

print('Done')

print(env.end_simulation())
env.close()
