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
--task_id " \
--list 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
# --vertical_steps 500 \
# --step_size 10 \



# export CUDA_VISIBLE_DEVICES=1

# python scripts/video_sample.py \
# checkpoints/2713al2i/ema_0.9999_500000.pt \
# --inference_mode autoreg \
# --step_size 10 \
# --T 100 \
# --indices 96 97 98 99
