import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import PIL
import imageio
import os
from tqdm.auto import tqdm

from improved_diffusion.script_util import (
    video_model_and_diffusion_defaults,
    create_video_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion import dist_util
from improved_diffusion.image_datasets import get_test_dataset


@torch.no_grad()
def infer_video(model, model_args, batch, max_T, obs_length):
    """
    batch has a shape of BxTxCxHxW where
    B: batch size
    T: video length
    CxWxH: image size
    """
    batch_size = len(batch)
    x0 = batch[:, :args.max_T].to(args.device)
    samples = [x0[:, :obs_length].cpu().numpy()]
    for i in range(0, model_args.T, max_T - obs_length):
        frame_indices = torch.arange(i, i + max_T).repeat((batch_size, 1)).to(x0.device)
        obs_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
        obs_mask[:, :obs_length] = 1
        latent_mask = 1 - obs_mask
        kinda_marg_mask = torch.zeros_like(batch[:, :max_T, :1, :1, :1])
        # Condition on the first 3 masks, generate the rest autoregressively.
        local_samples, attention_map = diffusion.p_sample_loop(
            model, x0.shape, clip_denoised=True,
            model_kwargs=dict(frame_indices=frame_indices,
                            x0=x0,
                            obs_mask=obs_mask,
                            latent_mask=latent_mask,
                            kinda_marg_mask=kinda_marg_mask),
            latent_mask=latent_mask,
            return_attn_weights=True)
        # Overwrite the observed part from the input video.
        local_samples[obs_mask.squeeze().bool()] = x0[obs_mask.squeeze().bool()]
        samples.append(local_samples[:, obs_length:].cpu().numpy())
        # Replace the first part of x (which we will condition on in the next iteration)
        # The rest of x0 is ignored
        x0[:, :obs_length] = local_samples[:, -obs_length:]
    return np.concatenate(samples, axis=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--batch_size", default=6)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Inference arguments
    parser.add_argument("--max_T", type=str, default=10,
                        help="Maximum length of the sequence that fits in the GPU memory.")
    parser.add_argument("--obs_length", type=str, default=3,
                        help="Number of frames to observe. It will observe this many frames from the beginning of max_T frames and predicts the rest.")
    args = parser.parse_args()

    drange = [-1, 1] # Range of the generated samples' pixel values

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = Namespace(**data["config"])
    # Load the model
    model, diffusion = create_video_model_and_diffusion(
        **args_to_dict(model_args, video_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
            state_dict
        )
    model = model.to(args.device)
    model.eval()
    # Load the test set
    dataset = get_test_dataset(data_path=model_args.data_path, T=model_args.T)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # Make the output directory (if does not exist)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving samples to {args.out_dir}")

    # Generate and store samples
    cnt = 0
    for batch, _ in tqdm(dataloader):
        batch = batch.to(args.device)
        recon = infer_video(model, model_args, batch, max_T=args.max_T, obs_length=args.obs_length)
        recon = (recon - drange[0]) / (drange[1] - drange[0])  * 255 # recon with pixel values in [0, 255]
        recon = recon.astype(np.uint8)
        for i in range(len(recon)):
            np.save(recon[i], os.path.join(args.out_dir, f"sample_{cnt + i:03d}.npy"))
        cnt += len(recon)
