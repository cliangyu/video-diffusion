# sleep 8h
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7


python command_launchers.py \
--command "python scripts/video_sample.py \
checkpoints/2713al2i/ema_0.9999_500000.pt \
--inference_mode autoreg \
--step_size 10 \
--T 1000 \
--task_id " \
--list 0 1 2 3 4 5 6 7 8 9 10 11 \
# --observed_frames x_t_minus_1 \
# --vertical_steps 500 \



# export CUDA_VISIBLE_DEVICES=1

# python scripts/video_sample.py \
# checkpoints/2713al2i/ema_0.9999_500000.pt \
# --inference_mode autoreg \
# --step_size 10 \
# --T 100 \
# --indices 96 97 98 99
