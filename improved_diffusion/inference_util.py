import numpy as np
import torch
import time
import lpips as lpips_metric


class InferenceStrategyBase:
    """ Inference strategies """
    def __init__(self, video_length: int, num_obs: int, max_frames: int, step_size: int, optimal_schedule_path=None):
        """ Inference strategy base class. It provides an iterator that returns
            the indices of the frames that should be observed and the frames that should be generated.

        Args:
            video_length (int): Length of the videos.
            num_obs (int): Number of frames that are observed from the beginning of the video.
            max_frames (int): Maximum number of frames (observed or latent) that can be passed to the model in one shot.
            step_size (int): Number of frames to generate in each step.
            optimal_schedule_path (str): If you want to run this inference strategy with an optimal schedule,
                pass in the path to the optimal schedule file here. Then, it will choose the observed
                frames based on the optimal schedule file. Otherwise, it will behave normally.
                The optimal schedule file is a .pt file containing a dictionary from step number to the
                list of frames that should be observed in that step.
        """
        print_str = f"Inferring using the inference strategy \"{self.typename}\""
        if optimal_schedule_path is not None:
            print_str += f", and the optimal schedule stored at {optimal_schedule_path}."
        else:
            print_str += "."
        print(print_str)
        self._video_length = video_length
        self._max_frames = max_frames
        self._done_frames = set(range(num_obs))
        self._obs_frames = list(range(num_obs))
        self._step_size = step_size
        self.optimal_schedule = None if optimal_schedule_path is None else torch.load(optimal_schedule_path)
        self._current_step = 0 # Counts the number of steps.

    def __next__(self):
        # Check if the video is fully generated.
        if self.is_done():
            raise StopIteration
        # Get the next indices from the function overloaded by each inference strategy.
        obs_frame_indices, latent_frame_indices = self.next_indices()
        # If using the optimal schedule, overwrite the observed frames with the optimal schedule.
        if self.optimal_schedule is not None:
            if self._current_step not in self.optimal_schedule:
                print(f"WARNING: optimal observations for prediction step #{self._current_step} was not found in the saved optimal schedule.")
                obs_frame_indices = []
            else:
                obs_frame_indices = self.optimal_schedule[self._current_step]
        # Type checks. Both observed and latent indices should be lists.
        assert isinstance(obs_frame_indices, list) and isinstance(latent_frame_indices, list)
        # Make sure the observed frames are either osbserved or already generated before
        for idx in obs_frame_indices:
            assert idx in self._done_frames, f"Attempting to condition on frame {idx} while it is not generated yet.\nGenerated frames: {self._done_frames}\nObserving: {obs_frame_indices}\nGenerating: {latent_frame_indices}"
        assert np.all(np.array(latent_frame_indices) < self._video_length)
        self._done_frames.update([idx for idx in latent_frame_indices if idx not in self._done_frames])
        self._current_step += 1
        return obs_frame_indices, latent_frame_indices

    def is_done(self):
        return len(self._done_frames) >= self._video_length

    def __iter__(self):
        self.step = 0
        return self

    def next_indices(self):
        raise NotImplementedError

    @property
    def typename(self):
        return type(self).__name__


class AdaptiveInferenceStrategyBase(InferenceStrategyBase):
    def __init__(self, videos, distance, *args, **kwargs):
        self.videos = videos
        if distance == 'mse':
            self.distance_func = lambda a, b: ((a - b)**2)
        elif distance == 'lpips':
            net = lpips_metric.LPIPS(net='alex', spatial=False).to(videos.device)
            self.distance_func = lambda a, b: net(a, b).flatten(start_dim=1).mean(dim=1).detach().cpu().numpy()
        else:
            raise NotImplementedError
        super().__init__(*args, **kwargs)

    def select_obs_indices(self, possible_next_indices, n):
        batch_size = len(self.videos)
        distances_from_start = []
        for i in range(1, len(possible_next_indices)):
            i1 = possible_next_indices[0]
            i2 = possible_next_indices[i]
            distances_from_start.append(self.distance_func(self.videos[:, i1], self.videos[:, i2]))
        distances_from_start = [np.zeros_like(distances_from_start[0])] + distances_from_start  # representing zero distance for first frame
        distances_from_start = np.stack(distances_from_start, axis=1)
        relative_distances = distances_from_start / distances_from_start.max(axis=1, keepdims=True)
        observe_when_distance_exceeds = np.linspace(0, 1, n)
        batch_indices = []
        for j in range(batch_size):
            indices = []
            for threshold in observe_when_distance_exceeds[::-1]:
                try:
                    first_index_exceeding_threshold = next(i for i in possible_next_indices if relative_distances[j, i] >= threshold and i not in indices)
                except StopIteration:
                    print('WARNING: couldn\'t find suitable index in adaptive index selection')
                    first_index_exceeding_threshold = next(i for i in possible_next_indices if i not in indices)
                indices.append(first_index_exceeding_threshold)
            batch_indices.append(indices)
        assert len(indices) == n
        return batch_indices

    def __next__(self):
        # Check if the video is fully generated.
        if self.is_done():
            raise StopIteration
        # Get the next indices from the function overloaded by each inference strategy.
        obs_frame_indices, latent_frame_indices = self.next_indices()
        # Type checks. Both observed and latent indices should be lists.
        assert isinstance(obs_frame_indices, list) and isinstance(latent_frame_indices, list)
        # Make sure the observed frames are either osbserved or already generated before
        for idx in np.array(obs_frame_indices).flatten():
            assert idx in self._done_frames, f"Attempting to condition on frame {idx} while it is not generated yet.\nGenerated frames: {self._done_frames}\nObserving: {obs_frame_indices}\nGenerating: {latent_frame_indices}"
        assert np.all(np.array(latent_frame_indices) < self._video_length)
        self._done_frames.update([idx for idx in latent_frame_indices if idx not in self._done_frames])
        self._current_step += 1
        return obs_frame_indices, [latent_frame_indices]*len(obs_frame_indices)


class AdaptiveAutoregressive(AdaptiveInferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_indices(self):
        first_idx = max(self._done_frames) + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._step_size, self._video_length)))
        possible_obs_indices = sorted(self._done_frames)[::-1]
        n_obs = self._max_frames - self._step_size
        obs_frame_indices = self.select_obs_indices(possible_obs_indices, n_obs)
        return obs_frame_indices, latent_frame_indices


class Autoregressive(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_indices(self):
        obs_frame_indices = sorted(self._done_frames)[-(self._max_frames - self._step_size):]
        first_idx = obs_frame_indices[-1] + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._step_size, self._video_length)))
        return obs_frame_indices, latent_frame_indices


class Independent(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_indices(self):
        obs_frame_indices = sorted(self._obs_frames)[-(self._max_frames - self._step_size):]
        first_idx = max(self._done_frames) + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._step_size, self._video_length)))
        return obs_frame_indices, latent_frame_indices


class ReallyIndependent(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_indices(self):
        obs_frame_indices = []
        first_idx = max(self._done_frames) + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._max_frames, self._video_length)))
        return obs_frame_indices, latent_frame_indices


class ExpPast(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_indices(self):
        cur_idx = max(self._done_frames) + 1
        distances_past = 2**np.arange(int(np.log2(cur_idx))) # distances from the observed frames (all in the past)
        obs_frame_indices = list(cur_idx - distances_past)
        latent_frame_indices = list(range(cur_idx, cur_idx + min(self._step_size, self._video_length)))
        # Observe more consecutive frames from the past if there is space left to reach max_frames.
        for i in range(1, cur_idx + 1):
            if len(obs_frame_indices) + len(latent_frame_indices) >= self._max_frames:
                break
            if cur_idx - i not in obs_frame_indices:
                obs_frame_indices.append(cur_idx - i)
        return obs_frame_indices, latent_frame_indices


class MixedAutoregressiveIndependent(InferenceStrategyBase):
    def next_indices(self):
        n_to_condition_on = self._max_frames - self._step_size
        n_autoreg_frames = n_to_condition_on // 2
        frames_to_condition_on = set(sorted(self._done_frames)[-n_autoreg_frames:])
        reversed_obs_frames = sorted(self._obs_frames)[::-1]
        for i in reversed_obs_frames:
            frames_to_condition_on.add(i)
            if len(frames_to_condition_on) == n_to_condition_on:
                break
        obs_frame_indices = sorted(frames_to_condition_on)
        first_idx = max(self._done_frames) + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._step_size, self._video_length)))
        return obs_frame_indices, latent_frame_indices


class HierarchyNLevel(InferenceStrategyBase):

    @property
    def N(self):
        raise NotImplementedError

    @property
    def sample_every(self):
        sample_every_on_level_1 = (self._video_length - len(self._obs_frames)) / (self._step_size-1)
        return int(sample_every_on_level_1 ** ((self.N-self.current_level)/(self.N-1)))

    def next_indices(self):
        if len(self._done_frames) == len(self._obs_frames):
            self.current_level = 1
            self.last_sampled_idx = max(self._obs_frames)

        n_to_condition_on = self._max_frames - self._step_size
        n_to_sample = self._step_size

        # select the grid of latent_frame_indices (shifting by 1 to avoid already-existing frames)
        idx = self.last_sampled_idx + self.sample_every
        if len([i for i in range(idx, self._video_length) if i not in self._done_frames]) == 0:
            # reset if there are no frames left to sample after idx
            self.current_level += 1
            self.last_sampled_idx = 0
            idx = min(i for i in range(self._video_length) if i not in self._done_frames) - 1 + self.sample_every
        if self.current_level == 1:
            latent_frame_indices = list(int(i) for i in np.linspace(max(self._obs_frames)+1, self._video_length-0.001, n_to_sample))
        else:
            latent_frame_indices = []
            while len(latent_frame_indices) < n_to_sample and idx < self._video_length:
                if idx not in self._done_frames:
                    latent_frame_indices.append(idx)
                    idx += self.sample_every
                elif idx in self._done_frames:
                    idx += 1

        # observe any frames in between latent frames
        obs_frame_indices = [i for i in range(min(latent_frame_indices), max(latent_frame_indices)) if i in self._done_frames]
        obs_before_and_after = n_to_condition_on - len(obs_frame_indices)

        if obs_before_and_after < 2: # reduce step_size if necessary to ensure conditioning before + after latents
            if self._step_size == 1:
                raise Exception('Cannot condition before and after even with step size of 1')
            sample_every = self.sample_every
            self._step_size -= 1
            result = self.next_indices()
            self._step_size += 1
            return result

        max_n_after = obs_before_and_after // 2
        # observe `n_after` frames afterwards
        obs_frame_indices.extend([i for i in range(max(latent_frame_indices)+1, self._video_length) if i in self._done_frames][:max_n_after])
        n_before = n_to_condition_on - len(obs_frame_indices)
        # observe `n_before` frames before...
        if self.current_level == 1:
            obs_frame_indices.extend(list(np.linspace(0, max(self._obs_frames)+0.999, n_before).astype(np.int32)))
        else:
            obs_frame_indices.extend([i for i in range(min(latent_frame_indices)-1, -1, -1) if i in self._done_frames][:n_before])

        self.last_sampled_idx = max(latent_frame_indices)

        return obs_frame_indices, latent_frame_indices

    @property
    def typename(self):
        return f"{super().typename}-{self.N}"


def get_hierarchy_n_level(n):
    class Hierarchy(HierarchyNLevel):
        N = n
    return Hierarchy

inference_strategies = {
    'autoreg': Autoregressive,
    'independent': Independent,
    'really-independent': ReallyIndependent,
    'exp-past': ExpPast,
    'mixed-autoreg-independent': MixedAutoregressiveIndependent,
    'hierarchy-2': get_hierarchy_n_level(2),
    'hierarchy-3': get_hierarchy_n_level(3),
    'hierarchy-4': get_hierarchy_n_level(4),
    'hierarchy-5': get_hierarchy_n_level(5),
    'adaptive-autoreg': AdaptiveAutoregressive,
}
