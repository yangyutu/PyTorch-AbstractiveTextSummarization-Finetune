export CUDA_VISIBLE_DEVICES="0"
export WANDB_CACHE_DIR="/mnt/d/MLData"
#pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
pretrained_model_name="facebook/bart-large"
pretrained_model_type="bart"
# for bart, max sequence length is 1024, lr 2e-3


pretrained_model_name="facebook/bart-large"
pretrained_model_type="bart"
python run_seq2seq_finetune_train.py \
--pretrained_model_type ${pretrained_model_type} \
--pretrained_model_name ${pretrained_model_name} \
--gpus 1 \
--max_epochs 2 \
--lr 2e-3 \
--lr_warm_up_steps 10000 \
--batch_size 16 \
--grad_accum 2 \
--num_workers 16 \
--truncate 128 \
--project_name abstract_text_summarization_finetune \
--default_root_dir ./experiments/logs

# pretrained_model_name="google/flan-t5-small"
# pretrained_model_name="t5-base"
# pretrained_model_type="t5"
# # for t5, max sequence length is 512, use precision 32 (use precision 16 might get NaN after some training steps)
# data_dir=/mnt/d/MLData/data/summarization/cnndm_bart/cnndm/diverse
# python run_bart_finetune_train.py \
# --pretrained_model_type ${pretrained_model_type} \
# --pretrained_model_name ${pretrained_model_name} \
# --gpus 1 \
# --max_epochs 2 \
# --lr 2e-3 \
# --lr_warm_up_steps 10000 \
# --precision 32 \
# --batch_size 8 \
# --grad_accum 4 \
# --num_workers 16 \
# --truncate 128 \
# --project_name abstract_text_summarization_finetune \
# --default_root_dir ./experiments/logs 