set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH='./download/models'
TASK_DATA_PATH='./download/task_data'
PYTHON_PATH='~/paddle_1.3/paddle_release_home/python/bin/python'

${PYTHON_PATH} -u run_classifier.py \
                   --use_cuda false \
                   --verbose true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --batch_size 32 \
                   --init_checkpoint ${MODEL_PATH}/params \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/lcqmc/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/lcqmc/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/lcqmc/test.tsv \
                   --vocab_path config/vocab.txt \
                   --checkpoints ./checkpoints \
                   --save_steps 1000 \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 1 \
                   --max_seq_len 128 \
                   --ernie_config_path config/ernie_config.json \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1

# Load model from ./download/models/params
# Final test result:
# [test evaluation] ave loss: 0.678016, ave acc: 0.575680, data_num: 12500, elapsed time: 2664.698558 s
