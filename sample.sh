export CUDA_VISIBLE_DEVICES=3,4,5,6,7

python command_launchers.py \
--command "python scripts/video_sample_full.py \
checkpoints/1k1hne30/model_500000.pt \
--inference_mode autoreg \
--step_size 7 \
--T 100 \
--observed_frames x_t_minus_1 \
--vertical_steps 500 \
--task_id " \
--list 3 4 5 6 7 8 9 10 11 \



# export CUDA_VISIBLE_DEVICES=1

# python scripts/video_sample_full.py \
# checkpoints/1k1hne30/model_500000.pt \
# --inference_mode autoreg \
# --step_size 7 \
# --T 100 \
# --observed_frames x_t_minus_1 \
# --vertical_steps 500 \
# --indices 99 98 97 96
