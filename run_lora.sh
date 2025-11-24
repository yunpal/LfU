
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"
MODE="LfU(LoRA)"  
DATASETS=("gsm8k_train")
BATCH_SIZE_LIST=(4)
ACCUM_STEPS_LIST=(4)
ALPHA_LIST=(0.1)
BETA_LIST=(5)


BASE_YAML_PATH="${SCRIPT_DIR}/yaml"
BASE_SAVE_PATH="${SCRIPT_DIR}/saves"
MERGE_YAML_PATH="${SCRIPT_DIR}/merge_yaml"
MERGE_OUT_BASE="${SCRIPT_DIR}/out"
RESULT_ROOT="$SCRIPT_DIR/results"
MODEL_NAME="meta-llama/Llama-3.1-8B"
TEMPLATE="alpaca"
EXPORT_SIZE=5
EXPORT_DEVICE="cpu"
EXPORT_LEGACY_FORMAT="false"



mkdir -p "$BASE_YAML_PATH" "$MERGE_YAML_PATH" "$MERGE_OUT_BASE" "$RESULT_ROOT" 

for dataset in "${DATASETS[@]}"; do
  for bsz in "${BATCH_SIZE_LIST[@]}"; do
    for accum in "${ACCUM_STEPS_LIST[@]}"; do
      for alpha in "${ALPHA_LIST[@]}"; do
        for beta in "${BETA_LIST[@]}"; do

          echo "🚀 Running training: mode=${MODE}, dataset=${dataset}, bsz=${bsz}, accum=${accum}, alpha=${alpha}, beta=${beta}"

          EXP_NAME="${dataset}_bsz${bsz}_acc${accum}_alpha${alpha}_beta${beta}"
          OUTPUT_DIR="${BASE_SAVE_PATH}/${EXP_NAME}"
          YAML_PATH="${BASE_YAML_PATH}/llama3_lora_${MODE}_10_${EXP_NAME}.yaml"

 
          cat <<EOF > "$YAML_PATH"
### model
model_name_or_path: ${MODEL_NAME}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_alpha: 16
lora_rank: 8
lora_target: all
mode: ${MODE}
beta: ${beta}
alpha: ${alpha}
start_layer: 0
finish_layer: 32

### dataset
dataset: ${dataset}
dataset_b: ${dataset}
template: ${TEMPLATE}
template_b: ${TEMPLATE}
cutoff_len: 2048
max_samples: 100000000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
seed: 42

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 1
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### wandb
report_to: wandb
run_name: ${MODE}_${EXP_NAME}

### train
per_device_train_batch_size: ${bsz}
gradient_accumulation_steps: ${accum}
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
EOF
          cd LLaMA-Factory
          llamafactory-cli train "$YAML_PATH"


          MERGE_YAML="${MERGE_YAML_PATH}/merge_${EXP_NAME}.yaml"
          MERGE_OUTPUT="${MERGE_OUT_BASE}/${EXP_NAME}/${dataset}"
          mkdir -p "$(dirname "$MERGE_OUTPUT")"

          cat <<EOF > "$MERGE_YAML"
### model
model_name_or_path: ${MODEL_NAME}
adapter_name_or_path: ${OUTPUT_DIR}
template: ${TEMPLATE}
trust_remote_code: true

### export
export_dir: ${MERGE_OUTPUT}
export_size: ${EXPORT_SIZE}
export_device: ${EXPORT_DEVICE}
export_legacy_format: ${EXPORT_LEGACY_FORMAT}
EOF
          llamafactory-cli export "$MERGE_YAML"

          done

        done
      done
    done
  done
done
