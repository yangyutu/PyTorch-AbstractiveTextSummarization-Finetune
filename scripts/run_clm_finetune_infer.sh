export CUDA_VISIBLE_DEVICES="0"
#pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
pretrained_model_name="facebook/bart-large"
pretrained_model_name="gpt2"
pretrained_model_name="gpt2-medium"
pretrained_model_type="gpt2"
python run_clm_finetune_infer.py \
--pretrained_model_name ${pretrained_model_name} \
--pretrained_model_type ${pretrained_model_type} \
--model_ckpt /mnt/d/MLData/Repos/PyTorch-AbstractiveTextSummarization-FineTune/ckpt/model-1fmlf2lx-v1/model.ckpt \
--gpus 1 \
--max_epochs 10 \
--batch_size 16 \
--grad_accum 4 \
--num_workers 16 \
--truncate 128 \
--num_beams 1 \
--project_name text_summarization_finetune \
--default_root_dir ./experiments/logs \
