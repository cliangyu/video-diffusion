import numpy as np
import torch
import time
import lpips as lpips_metric


def not_spatial_average(in_tens, keepdim=True):
    B, C, H, W = in_tens.shape
    reshaped = in_tens.view(B, C*H*W, 1, 1) if keepdim else in_tens.view(B, C*H*W)
    return reshaped / (H*W)**0.5

class LpipsEmbedder(lpips_metric.LPIPS):
    def scale_by_proj_weights(self, proj, x):
        conv = proj.model[-1]
        proj_weights = conv.weight
        return (proj_weights**0.5) * x

    def forward(self, x):
        outs = self.net.forward(self.scaling_layer(x))
        feats = {}
        for kk in range(self.L):
            feats[kk] = lpips_metric.normalize_tensor(outs[kk])
        # TODO not spatial average
        res = [not_spatial_average(self.scale_by_proj_weights(self.lins[kk], feats[kk]), keepdim=True) for kk in range(self.L)]
        return torch.cat(res, dim=1)


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
    def __init__(self, distance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance = distance

    def embed(self, indices):
        if self.distance == 'l2':
            embs = [self.videos[:, i] for i in indices]
        elif self.distance == 'lpips':
            net = LpipsEmbedder(net='alex', spatial=False).to(self.videos.device)
            embs = [net(self.videos[:, i]) for i in indices]
        else:
            raise NotImplementedError
        return torch.stack(embs, dim=1)

    def set_videos(self, videos):
        self.videos = videos

    def select_obs_indices(self, possible_next_indices, n, always_selected=(0,)):
        B = len(self.videos)
        embs = self.embed(possible_next_indices)
        batch_selected_indices = []
        for b in range(B):
            min_distances_from_selected = [np.inf for _ in possible_next_indices]
            selected_indices = [possible_next_indices[always_selected[0]]]
            selected_embs = [embs[b, always_selected[0]]]  # always begin with next possible index
            for i in range(1, n):
                # update min_distances_from_selected
                for f, dist in enumerate(min_distances_from_selected):
                    dist_to_newest = ((selected_embs[-1] - embs[b][f])**2).sum().cpu().item()
                    min_distances_from_selected[f] = min(dist, dist_to_newest)
                if i < len(always_selected):
                    best_index = always_selected[i]
                else:
                    # select one with maximum min_distance_from_selected
                    best_index = np.argmax(min_distances_from_selected)
                selected_indices.append(possible_next_indices[best_index])
                selected_embs.append(embs[b, best_index])
            batch_selected_indices.append(selected_indices)
        return batch_selected_indices

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
        possible_obs_indices = sorted(self._done_frames)
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


class AdaptiveHierarchyNLevel(AdaptiveInferenceStrategyBase, HierarchyNLevel):

    def next_indices(self):
        """
        Certainly not mainly copy-pasted from HierarchyNLevel...
        """
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

        # observe closest frame before and after latent frames
        i = min(latent_frame_indices)
        while i not in self._done_frames:
            i -= 1
        obs_frame_indices.append(i)
        # actually closest two before...
        i -= 1
        while i not in self._done_frames:
            i -= 1
        obs_frame_indices.append(i)
        i = max(latent_frame_indices)
        while i not in self._done_frames and i < self._video_length:
            i += 1
        if i < self._video_length:
            obs_frame_indices.append(i)

        # select which other frames to use adaptively
        possible_next_indices = list(self._done_frames)
        always_selected = [possible_next_indices.index(i) for i in obs_frame_indices]
        print('ALWAYS SELECTED', obs_frame_indices)
        obs_frame_indices = self.select_obs_indices(
            possible_next_indices=possible_next_indices, n=n_to_condition_on, always_selected=always_selected,
        )

        self.last_sampled_idx = max(latent_frame_indices)
        return obs_frame_indices, latent_frame_indices


def get_hierarchy_n_level(n):
    class Hierarchy(HierarchyNLevel):
        N = n
    return Hierarchy

def get_adaptive_hierarchy_n_level(n):
    class AdaptiveHierarchy(AdaptiveHierarchyNLevel):
        N = n
    return AdaptiveHierarchy


class GoalDirectedInferenceStrategy(HierarchyNLevel):

    def __next__(self):
        obs_frame_indices, latent_frame_indices = super().__next__()
        if self._done_frames == self._obs_frames:  # then it is the first step
            last_frame = self._video_length-1
            assert last_frame in latent_frame_indices
            latent_frame_indices.remove(last_frame)
            obs_frame_indices.append(last_frame)


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
    'adaptive-hierarchy-2': get_adaptive_hierarchy_n_level(2),
    'adaptive-hierarchy-3': get_adaptive_hierarchy_n_level(3),
}
