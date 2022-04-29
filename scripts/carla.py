from invertedai_simulate.utils import Res, SensorSettings, Resolution, ClientSideBoundingBoxes
from invertedai_simulate.interface import IAIEnv, ServerTimeoutError
import numpy as np
import time
import cv2
import imageio_ffmpeg
import imageio
import matplotlib.pyplot as plt
import argparse
IN_COLAB = False

def_res = Resolution(128, 128)
scale = 1

parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=str, help="Where to save the generated videos/coords.")
parser.add_argument('--port', type=int, default=5555, help="5555 is the server with other cars. Currently there are no other servers")
args = parser.parse_args()

sensors_dict = {
        'front-cam': {
            'sensor_type': 'camera',
            'camera_type': 'rgb-camera',
            'bounding_box': False,
            'show_bounding_boxes': False,
            'world_sensor': False,
            'resolution': def_res,
            'location': SensorSettings.Location(x=0, z=2.8, y=0),
            'rotation': SensorSettings.Rotation(yaw=0, roll=0, pitch=0),
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
    return {'images':[], 'coords': []}
frames = reset_frames()

repeats = 2
T = 10
walltime = time.time()
for i in range(0, T*repeats):
    action = info['expert_action']
    obs, reward, done, info = env.step(action)
    frames['images'].append(obs['sensor_data']['front-cam']['image'])
    x, y, z = obs['compact_vector'][:3]
    frames['coords'].append((x, y, z))

    if i % T == T-1:
        imageio.mimwrite(f'datasets/carla/gen-videos/video_{i//T}.mp4', frames['images'], fps=10, quality=7)
        coords = np.array(frames['coords'])
        np.save(f'datasets/carla/gen-videos/coords_{i//T}.npy', coords)
        frames = reset_frames()
        print(f'generated {T} frames in {time.time()-walltime} seconds')
        walltime = time.time()

print('Done')

print(env.end_simulation())
env.close()
