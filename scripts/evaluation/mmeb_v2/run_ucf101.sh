#!/usr/bin/env bash
# Run UCF101 video classification evaluation with Qwen3-VL-Embedding-2B

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MODEL_NAME="Qwen/Qwen3-VL-Embedding-2B"
MODEL_BASENAME=$(basename "$MODEL_NAME")
BATCH_SIZE=4
DATA_BASEDIR="data/evaluation/mmeb_v2"
DATA_CONFIG_PATH="scripts/evaluation/mmeb_v2/ucf101_only.yaml"
OUTPUT_PATH="results/evaluation/mmeb_v2/$MODEL_BASENAME/video/"

echo "================================================="
echo "UCF101 Evaluation with $MODEL_NAME"
echo "Output: $OUTPUT_PATH"
echo "================================================="

mkdir -p "$OUTPUT_PATH"

CUDA_VISIBLE_DEVICES=0 pixi run python -m src.evaluation.mmeb_v2.eval_embedding \
    --normalize true \
    --per_device_eval_batch_size $BATCH_SIZE \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_config "$DATA_CONFIG_PATH" \
    --encode_output_path "$OUTPUT_PATH" \
    --data_basedir "$DATA_BASEDIR"

if [ $? -eq 0 ]; then
    echo "✅ UCF101 evaluation completed successfully."
    echo "Results saved to: $OUTPUT_PATH"
    cat "$OUTPUT_PATH/UCF101_score.json" 2>/dev/null
else
    echo "❌ UCF101 evaluation failed."
    exit 1
fi
