export CUDA_VISIBLE_DEVICES="0"
export WANDB_CACHE_DIR="/mnt/d/MLData"
#pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
pretrained_model_name="gpt2-medium"
pretrained_model_type="gpt"

# for gpt2, max sequence length is 512, use precision 32 (use precision 16 might get NaN after some training steps)
python run_clm_finetune_train.py \
--pretrained_model_type ${pretrained_model_type} \
--pretrained_model_name ${pretrained_model_name} \
--gpus 1 \
--max_epochs 2 \
--lr 2e-3 \
--lr_warm_up_steps 10000 \
--precision 16 \
--batch_size 4 \
--grad_accum 4 \
--num_workers 16 \
--truncate 128 \
--project_name abstract_text_summarization_finetune \
--default_root_dir ./experiments/logs 