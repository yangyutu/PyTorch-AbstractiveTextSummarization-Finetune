# Abstractive Text Summarization Finetune

This repo contains the examples to finetune pretrained Transformers for abstractive text summarization. We utilize pytorch-lightning as the training framework, and HuggingFace Transformer as the model architectures.

We mainly consider two types of model architectures for abstractive text summarization
 - Encoder-Decoder Transformers (e.g., Bart and T5)
 - Decoder only Transformers (e.g., GPT-2)

## How to run

### Finetuning training

For seq2seq finetuning Bart model, run the following

```
export CUDA_VISIBLE_DEVICES="0"
export WANDB_CACHE_DIR="/mnt/d/MLData"
pretrained_model_name="facebook/bart-large"
pretrained_model_type="bart"
python run_bart_finetune_train.py \
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

```

For seq2seq finetuning t5 model, run the following
Note that for t5, max sequence length is 512, use precision 32 (use precision 16 might get NaN after some training steps)

```
export pretrained_model_type="t5"
python run_bart_finetune_train.py \
--pretrained_model_type ${pretrained_model_type} \
--pretrained_model_name ${pretrained_model_name} \
--gpus 1 \
--max_epochs 2 \
--lr 2e-3 \
--lr_warm_up_steps 10000 \
--precision 32 \
--batch_size 8 \
--grad_accum 4 \
--num_workers 16 \
--truncate 128 \
--project_name abstract_text_summarization_finetune \
--default_root_dir ./experiments/logs 
```

### Inference

### Zero-shot evaluation

facebook/bart-large no finetune
evalution on cnn/dm test split
{'rouge-1': {'r': 0.549681154365426, 'p': 0.31894798491288195, 'f': 0.39527814571759634}, 'rouge-2': {'r': 0.24119011060513393, 'p': 0.12342848717231476, 'f': 0.15859760141602058}, 'rouge-l': {'r': 0.5033519772533492, 'p': 0.2917140899959914, 'f': 0.3616936114853321}}

{'rouge-1': {'r': 0.502183855105744, 'p': 0.272072077203377, 'f': 0.34548660799629516}, 'rouge-2': {'r': 0.214393492152304, 'p': 0.10415730007709417, 'f': 0.13640398627419068}, 'rouge-l': {'r': 0.46584268300328907, 'p': 0.25205800362899533, 'f': 0.320230113001226}}
{'rouge1': 0.36351439639780003, 'rouge2': 0.15842926351282907, 'rougeL': 0.22778493147783768, 'rougeLsum': 0.2278240879965529}


### Fine-tuned model evaluation

facebook/bart-large no finetune
evalution on cnn/dm test split
{'rouge-1': {'r': 0.549681154365426, 'p': 0.31894798491288195, 'f': 0.39527814571759634}, 'rouge-2': {'r': 0.24119011060513393, 'p': 0.12342848717231476, 'f': 0.15859760141602058}, 'rouge-l': {'r': 0.5033519772533492, 'p': 0.2917140899959914, 'f': 0.3616936114853321}}

official facebook/bart-large-cnn
{'rouge-1': {'r': 0.4796892456870575, 'p': 0.3735731019790997, 'f': 0.4113638761364382}, 'rouge-2': {'r': 0.20889529982588406, 'p': 0.16071571383842673, 'f': 0.17647569935797405}, 'rouge-l': {'r': 0.44430798082241246, 'p': 0.3461935381771986, 'f': 0.38113134121078657}}

my fine-tune version
{'rouge-1': {'r': 0.5126184563211834, 'p': 0.360592121475519, 'f': 0.41479543521156514}, 'rouge-2': {'r': 0.22943074772174232, 'p': 0.1524198085528376, 'f': 0.17798423730136556}, 'rouge-l': {'r': 0.47732220929004227, 'p': 0.3355634913157298, 'f': 0.3860966896120778}}


Using huggingface own dataset

facebook/bart-large

{'rouge-1': {'r': 0.502183855105744, 'p': 0.272072077203377, 'f': 0.34548660799629516}, 'rouge-2': {'r': 0.214393492152304, 'p': 0.10415730007709417, 'f': 0.13640398627419068}, 'rouge-l': {'r': 0.46584268300328907, 'p': 0.25205800362899533, 'f': 0.320230113001226}}
{'rouge1': 0.36351439639780003, 'rouge2': 0.15842926351282907, 'rougeL': 0.22778493147783768, 'rougeLsum': 0.2278240879965529}

my own finetuned bart

truncate at 128 during train
{'rouge-1': {'r': 0.44529018994755637, 'p': 0.37294873095016023, 'f': 0.3974957048418661}, 'rouge-2': {'r': 0.2036975347152083, 'p': 0.16548030587627488, 'f': 0.1776059811744489}, 'rouge-l': {'r': 0.4175821256104588, 'p': 0.34969885747090357, 'f': 0.3727460707790733}}
{'rouge1': 0.4210376346449135, 'rouge2': 0.20051787410466726, 'rougeL': 0.29061919255992236, 'rougeLsum': 0.2906113309774484}


flan-t5-base zero-shot
{'rouge-1': {'r': 0.3574858911230986, 'p': 0.3894666706292488, 'f': 0.360590412823733}, 'rouge-2': {'r': 0.1539909921627613, 'p': 0.16712079402368288, 'f': 0.1535680861455767}, 'rouge-l': {'r': 0.3340371094808994, 'p': 0.36328641305131393, 'f': 0.3366720817043174}}
{'rouge1': 0.38513287137473934, 'rouge2': 0.17588147540472285, 'rougeL': 0.27126519610753586, 'rougeLsum': 0.2712792118841236}


t5-base zero-shot
{'rouge-1': {'r': 0.328392461444821, 'p': 0.35254029164678985, 'f': 0.3327791974833028}, 'rouge-2': {'r': 0.13389104873322732, 'p': 0.14146338790794133, 'f': 0.13378804306770192}, 'rouge-l': {'r': 0.3074218542513367, 'p': 0.3299819163816069, 'f': 0.3115174368593911}}
{'rouge1': 0.3884620096102989, 'rouge2': 0.17548528757963683, 'rougeL': 0.27323186221569395, 'rougeLsum': 0.2731558517380708}

t5-base my own finetune
{'rouge-1': {'r': 0.3731784029902917, 'p': 0.3876612721460803, 'f': 0.37217602028647057}, 'rouge-2': {'r': 0.1593356630493261, 'p': 0.16603060291155247, 'f': 0.15812325511090364}, 'rouge-l': {'r': 0.3507091090356709, 'p': 0.36429129540370253, 'f': 0.34974957748965935}}
{'rouge1': 0.39921618401058956, 'rouge2': 0.18038736736617694, 'rougeL': 0.275406257528953, 'rougeLsum': 0.2753755169548109}


gpt2 base zero-shot

{'rouge-1': {'r': 0.2247027388001004, 'p': 0.20489069003673158, 'f': 0.20330664801520082}, 'rouge-2': {'r': 0.03695070775532497, 'p': 0.026351623967469047, 'f': 0.028573094556441206}, 'rouge-l': {'r': 0.2072931740559499, 'p': 0.18906569196683648, 'f': 0.1875076870087442}}
{'rouge1': 0.20728679056964092, 'rouge2': 0.03583599100935715, 'rougeL': 0.13388462716465888, 'rougeLsum': 0.1542299934933517}


gpt2 fine-tuned
{'rouge-1': {'r': 0.3678002390587857, 'p': 0.24308421267414043, 'f': 0.286140428000665}, 'rouge-2': {'r': 0.09028252864149315, 'p': 0.04655261260679583, 'f': 0.05969153931678913}, 'rouge-l': {'r': 0.3485751865245865, 'p': 0.22985542699529812, 'f': 0.2708258956951285}}
{'rouge1': 0.2831426843004533, 'rouge2': 0.07432778909202704, 'rougeL': 0.169747234483531, 'rougeLsum': 0.16980411379765684}


gpt2-medium zero-shot
{'rouge-1': {'r': 0.2564629388361701, 'p': 0.2216070796346227, 'f': 0.2259239869396991}, 'rouge-2': {'r': 0.04759910281276827, 'p': 0.03309606107612852, 'f': 0.03646539062894255}, 'rouge-l': {'r': 0.23504664228432293, 'p': 0.20297481796880845, 'f': 0.2069509191393846}}
{'rouge1': 0.23443871226323876, 'rouge2': 0.047221976261152, 'rougeL': 0.14612735800435894, 'rougeLsum': 0.17480366960587834}

gpt2-medium fine tune
{'rouge-1': {'r': 0.39451350431803656, 'p': 0.2497381576988981, 'f': 0.2989689577773459}, 'rouge-2': {'r': 0.10295262162827354, 'p': 0.051806940467411404, 'f': 0.06697710149875646}, 'rouge-l': {'r': 0.3730280054569123, 'p': 0.23562617513240916, 'f': 0.28233314101178353}}
{'rouge1': 0.29613467992065634, 'rouge2': 0.08181973331420882, 'rougeL': 0.17837232706949802, 'rougeLsum': 0.178565779863864}