import numpy as np


class InferenceStrategyBase:
    """ Inference strategies """
    def __init__(self, video_length: int, num_obs: int, max_T: int, step_size: int):
        """ Inference strategy base class. It provides an iterator that returns
            the indices of the frames that should be observed and the frames that should be generated.

        Args:
            video_length (int): Length of the videos.
            num_obs (int): Number of frames that are observed from the beginning of the video.
            max_T (int): Maximum number of frames (observed or latent) that can be passed to the model in one shot.
            step_size (int): Number of frames to generate in each step.
        """
        self._video_length = video_length
        self._max_T = max_T
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
        obs_frame_indices = sorted(self._done_frames)[-(self._max_T - self._step_size):]
        first_idx = obs_frame_indices[-1] + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._step_size, self._video_length)))
        return obs_frame_indices, latent_frame_indices


class Independent(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def next_indices(self):
        obs_frame_indices = sorted(self._obs_frames)[-(self._max_T - self._step_size):]
        first_idx = max(self._done_frames) + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._step_size, self._video_length)))
        return obs_frame_indices, latent_frame_indices


class ExpPast(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def next_indices(self):
        cur_idx = max(self._done_frames) + 1
        distances_past = 2**np.arange(int(np.log2(cur_idx))) # distances from the observed frames (all in the past)
        obs_frame_indices = list(cur_idx - distances_past)
        latent_frame_indices = list(range(cur_idx, cur_idx + min(self._step_size, self._video_length)))
        for i in range(1, cur_idx + 1):
            if len(obs_frame_indices) + len(latent_frame_indices) >= self._max_T:
                break
            if cur_idx - i not in obs_frame_indices:
                obs_frame_indices.append(cur_idx - i)
        return obs_frame_indices, latent_frame_indices