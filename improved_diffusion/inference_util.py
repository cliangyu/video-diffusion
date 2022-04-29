import numpy as np


class InferenceStrategyBase:
    """ Inference strategies """
    def __init__(self, video_length: int, num_obs: int, max_frames: int, step_size: int):
        """ Inference strategy base class. It provides an iterator that returns
            the indices of the frames that should be observed and the frames that should be generated.

        Args:
            video_length (int): Length of the videos.
            num_obs (int): Number of frames that are observed from the beginning of the video.
            max_frames (int): Maximum number of frames (observed or latent) that can be passed to the model in one shot.
            step_size (int): Number of frames to generate in each step.
        """
        self._video_length = video_length
        self._max_frames = max_frames
        self._done_frames = set(range(num_obs))
        self._obs_frames = list(range(num_obs))
        self._step_size = step_size
    
    def __next__(self):
        # Check if the video is fully generated.
        if self.is_done():
            raise StopIteration
        # Get the next indices from the function overloaded by each inference strategy.
        obs_frame_indices, latent_frame_indices = self.next_indices()
        # Type checks. Both observed and latent indices should be lists.
        assert isinstance(obs_frame_indices, list) and isinstance(latent_frame_indices, list)
        # Make sure the observed frames are either osbserved or already generated before
        for idx in obs_frame_indices:
            assert idx in self._done_frames, f"Attempting to condition on frame {idx} while it is not generated yet.\nGenerated frames: {self._done_frames}\nObserving: {obs_frame_indices}\nGenerating: {latent_frame_indices}"
        assert np.all(np.array(latent_frame_indices) < self._video_length)
        self._done_frames.update([idx for idx in latent_frame_indices if idx not in self._done_frames])
        return obs_frame_indices, latent_frame_indices
    
    def is_done(self):
        return len(self._done_frames) >= self._video_length
    
    def __iter__(self):
        return self
        
    def next_indices(self):
        raise NotImplementedError


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
        sample_every_on_level_1 = (self._video_length - len(self._obs_frames)) / self._step_size
        return int(sample_every_on_level_1 ** ((self.N-self.current_level)/(self.N-1)))

    def next_indices(self):
        if len(self._done_frames) == len(self._obs_frames):
            self.current_level = 1
            self.last_sampled_idx = max(self._obs_frames)

        n_to_condition_on = self._max_frames - self._step_size
        n_to_sample = self._step_size

        # select the grid of latent_frame_indices (shifting by 1 to avoid already-existing frames)
        latent_frame_indices = []
        idx = self.last_sampled_idx + self.sample_every
        if idx >= self._video_length:
            self.current_level += 1
            self.last_sampled_idx = 0
            idx = min(i for i in range(self._video_length) if i not in self._done_frames) - 1 + self.sample_every

        while len(latent_frame_indices) < n_to_sample and idx < self._video_length:
            if idx not in self._done_frames:
                latent_frame_indices.append(idx)
                idx +=  self.sample_every
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
            print('could not condition before and after with previous step size, recursing with step size', self._step_size)
            result = self.next_indices()
            self._step_size += 1
            return result

        max_n_after = obs_before_and_after // 2
        # observe `n_after` frames afterwards
        obs_frame_indices.extend([i for i in range(max(latent_frame_indices)+1, self._video_length) if i in self._done_frames][:max_n_after])
        n_before = n_to_condition_on - len(obs_frame_indices)
        # observe `n_before` frames before...
        if self.current_level == 1:
            obs_frame_indices.extend(list(range(max(self._obs_frames), -1, -int(max(self._obs_frames)/(n_before-1)))))
        else:
            obs_frame_indices.extend([i for i in range(min(latent_frame_indices)-1, -1, -1) if i in self._done_frames][:n_before])

        self.last_sampled_idx = max(latent_frame_indices)

        return obs_frame_indices, latent_frame_indices


class Hierarchy2Level(InferenceStrategyBase):
    def next_indices(self):
        n_to_condition_on = self._max_frames - self._step_size
        n_to_sample = self._step_size
        if len(self._done_frames) == len(self._obs_frames):
            obs_frame_indices = [int(i) for i in np.linspace(min(self._obs_frames), max(self._obs_frames), n_to_condition_on)]
            latent_frame_indices = [int(i) for i in np.linspace(max(self._obs_frames)+1, self._video_length-1, n_to_sample)]
        else:
            latent_frame_indices = []
            idx = max(self._obs_frames)
            while True:
                idx += 1
                if idx >= self._video_length or len(latent_frame_indices) == n_to_sample:
                    break
                elif idx not in self._done_frames:
                    latent_frame_indices.append(idx)
            obs_frame_indices = []
            for idx in range(min(latent_frame_indices)+1, max(latent_frame_indices)):
                # observe indices in the middle of the latents
                if idx not in latent_frame_indices:
                    obs_frame_indices.append(idx)
            remaining_to_condition_on = n_to_condition_on - len(obs_frame_indices)
            n_cond_after = remaining_to_condition_on // 2
            n_cond_before = remaining_to_condition_on - n_cond_after
            obs_frame_indices.extend(range(min(latent_frame_indices)-n_cond_before, min(latent_frame_indices)))
            idx_after = max(latent_frame_indices)
            while len(obs_frame_indices) < n_to_condition_on:
                idx_after += 1
                if idx_after >= self._video_length:
                    break
                elif idx_after in self._done_frames:
                    obs_frame_indices.append(idx_after)
        return obs_frame_indices, latent_frame_indices


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
    'hierarchy-2': Hierarchy2Level,
    'hierarchy-2-new': get_hierarchy_n_level(2),
    'hierarchy-3': get_hierarchy_n_level(3),
    'hierarchy-4': get_hierarchy_n_level(4),
}
