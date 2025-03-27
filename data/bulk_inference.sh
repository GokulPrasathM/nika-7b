#! /bin/bash
# iterate over shard_index for 0 to 6, including 0 and 6
for shard_index in {0..5}; do
    srun --job-name=bulk \
        --nodes=1 \
        --account=marlowe-m000027 \
        --partition=beta \
        --gpus-per-task=2 \
        --mem=256G \
        --time=3-00:00:00 \
        python data/scripts/bulk_inference.py \
        --shard_index ${shard_index} \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        > data/scripts/log/r1_qwen_7b_${shard_index}.txt 2>&1 &
done

# iterate over shard_index for 0 to 6, including 0 and 6
for shard_index in {0..5}; do
    srun --job-name=bulk \
        --nodes=1 \
        --account=marlowe-m000027 \
        --partition=beta \
        --gpus-per-task=2 \
        --mem=256G \
        --time=3-00:00:00 \
        python data/scripts/bulk_inference.py \
        --shard_index ${shard_index} \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        > data/scripts/log/r1_qwen_14b_${shard_index}.txt 2>&1 &
done
