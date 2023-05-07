export CUDA_VISIBLE_DEVICES="0"
pretrained_model_name="yangyutu/bart-large-cnndm_ft"

python tasks/run_seq2seq_inference.py \
--pretrained_model_name ${pretrained_model_name} \
--article_truncate 512 \
--precision 16
