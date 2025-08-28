set -x
# export CUDA_VISIBLE_DEVICES=0
ENGINE=${1:-vllm}
export http_proxy=http://192.168.32.28:18000 
export https_proxy=http://192.168.32.28:18000
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export WANDB_API_KEY="6c32f101d0fd2e564837211b0be254f9f0c89835"
HF_MODEL_PATH=/media/vlm_model/Qwen2.5-VL-7B-Instruct
DIST_CKPT_PATH=/media/ckp/Joy/verl_dist/test
# python scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH
# SAVE_CHECKPOINT_DIR="/media/ckp/Joy/nnode_test2/verl/examples/grpo_trainer/outputs/change_kl_from_scarch_debug"
# convert HF model to verl format
# PYTHONPATH=/mnt/cluster/xiaojunjie/code/verl/:/mnt/cluster/xiaojunjie/code/Megatron/ \
# python /mnt/cluster/xiaojunjie/code/verl/scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH --use_cpu_initialization
# exit
# train_path=/media/luckkky/code/WFM_SYS/robor1/verl/data/geo3k/train.parquet
# train_path=/media/datasets/physGame/result/bak/violates_physics_commonsense_with_metadata.parquet
# test_path=/media/datasets/physGame/result/bak/violates_physics_commonsense_with_metadata.parquet
# train_path=/media/datasets/physGame/result/bak/train_split.parquet
# test_path=/media/ckp/Shuai/EgoLife/work/EmbodyProject/WFM_SYS/robovlm_eval/data/erqa_cot_for_verl_fix_egolife.parquet

train_cosmos=/media/luckkky/luckkky/code/verl_humanoid/rl_data/merged_valid_converted_fixed_fps2_pixels_x2_cosmos_align_fixed.parquet
train_egomcq=/media/luckkky/luckkky/code/verl_humanoid/rl_data/egomcq_train_fixed_align_fixed.parquet
train_robopoint=/media/datasets/vlm_data/r1_data/robopoint-ver1-train.parquet
train_scannet=/media/datasets/VLM-3R-DATA/vsibench_train/rl/train_4096_only_coherent_add_video_info.parquet



test_cosmos=/media/luckkky/luckkky/code/verl_humanoid/rl_data/cosmos_cot_with_extra_info_add_video_info_align_fixed.parquet
test_egomcq=/media/luckkky/luckkky/code/verl_humanoid/rl_data/egomcq_test_fixed.parquet
test_robopoint=/media/datasets/vlm_data/r1_data/robopoint-ver1-test.parquet 
test_scannet=/media/datasets/VLM-3R-DATA/vsibench_train/rl/test_512_only_coherent_add_video_info.parquet

train_path=/media/luckkky/luckkky/code/verl_humanoid/rl_data/cosmos_cot_with_extra_info_add_video_info_align_fixed.parquet #/media/luckkky/luckkky/code/verl_humanoid/verl/data/self_data/merged_cosmos_phy_egomcq.parquet
test_path=/media/luckkky/luckkky/code/verl_humanoid/rl_data/cosmos_cot_with_extra_info_add_video_info_align_fixed.parquet
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=5.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.default_local_dir=/media/ckp/Joy/verl_dist/test_dist_tste1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen2_5_vl_32b_megatron' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    # trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    
    # trainer.default_local_dir=$SAVE_CHECKPOINT_DIR \