# sleep 8h
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7


python command_launchers.py \
--command "python scripts/video_sample_full.py \
checkpoints/1k1hne30/ema_0.9999_500000.pt \
--inference_mode autoreg \
--step_size 7 \
--T 100 \
--batch_size 2 \
--observed_frames x_t_minus_1 \
--use_gradient_method \
--task_id " \
--list 36 37 38 39 40 41 42 43 44 45 46 47 48 49
# --vertical_steps 500 \
# --step_size 10 \



# export CUDA_VISIBLE_DEVICES=1

# python scripts/video_sample.py \
# checkpoints/2713al2i/ema_0.9999_500000.pt \
# --inference_mode autoreg \
# --step_size 10 \
# --T 100 \
# --indices 96 97 98 99
