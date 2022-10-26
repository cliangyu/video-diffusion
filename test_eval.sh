export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

wandb_id=2713al2i # baseline
# wandb_id=2mtsk9sv # baseline by authors
inference_mode=autoreg
step_size=10
T=1000

command="python scripts/video_sample_full.py \
checkpoints/${wandb_id}/ema_0.9999_500000.pt \
--inference_mode ${inference_mode} \
--step_size ${step_size} \
--T ${T} \
--observed_frames x_t_minus_1 \
"

# command="python scripts/video_sample.py \
# checkpoints/${wandb_id}/ema_0.9999_500000.pt \
# --inference_mode ${inference_mode} \
# --step_size ${step_size} \
# --T ${T} \
# "

python command_launchers.py \
--command "${command} --task_id " \
--list 0 1 2 3 4 5 6 7 8 9 10 11

eval "${command} --indices "\
96 97 98 99

python scripts/video_eval.py \
# --eval_dir "results/${wandb_id}/ema_0.9999_500000/${inference_mode}_None_${step_size}_${T}_36"
--eval_dir "results/${wandb_id}/ema_0.9999_500000/${inference_mode}_None_${step_size}_${T}_36_full"
