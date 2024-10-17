# 需要下载的文件

1. BLIP的权重，下载位置为：./BLIP/chatir_weights.ckpt
   下载链接：https://drive.google.com/file/d/1pW-Rt51yfsocK3Qnyom4hcxA22I7b-nL/view?usp=drive_link

```
.
└── BLIP
    ├── BLIP.gif
    ├── chatir_weights.ckpt

    ... omit other files
```


2. 图片召回相关权重

   下载链接为：https://drive.google.com/drive/folders/1DG6pYotka2RMtcDJcNoRgl6Z2IIGo3sX?usp=share_link

使用faiss构建的图片向量数据库，下载位置为./checkpoints/blip_faiss.index

图片向量数据库中图片的位置文件，下载位置为./checkpoints/id2image.pickle

图片向量数据库中图片的向量，下载位置为./checkpoints/blip_image_embedding.pickle

```/checkpoints$ tree
.
└── checkpoints
    ├── blip_faiss.index
    ├── id2image.pickle
    ├── blip_image_embedding.pickle
```



3. PLlava权重，权重位置为./MODELS/

```shell
git clone https://huggingface.co/ermu2001/pllava-7b
```
或者手动下载https://huggingface.co/ermu2001/pllava-7b/tree/main 中所有的文件
```
wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/train.json

git clone https://huggingface.co/nlphuji/flickr30k
.
└── MODELS
    └── pllava-7b
        ├── added_tokens.json
        ├── config.json
        ├── generation_config.json
        ├── gitattributes
        ├── model-00001-of-00003.safetensors
        ├── model-00002-of-00003.safetensors
        ├── model-00003-of-00003.safetensors
        ├── model.safetensors.index.json
        ├── preprocessor_config.json
        ├── processor_config.json
        ├── README.md
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── tokenizer.model
```

4. 下载图片到 /playground/data/css_data/unlabeled2017
* DownLoad the COCO dataset from the website. 

* Then unzip the unlabeled2017 file

```shell
wget http://images.cocodataset.org/zips/unlabeled2017.zip 
unzip unlabeled2017.zip
rm unlabeled2017.zip
```

```
.
└── playground
    └── data
       └── css_data
          └── unlabeled2017
              00000003323.jpg
              00000003324.jpg
              ... omit other images
```





# 环境配置

python环境依赖包

```shell
pip install transformers einops accelerate scikit-learn pandas faiss-cpu matplotlib seaborn trl peft flash-attn bitsandbytes opencv-python sentencepiece protobuf timm fairscale
```



### 整个流程的代码

1. 使用Pllava作为提问模型

   代码：step6.1all_process_pllava.ipynb

2. 使用ChatGPT-4o作为提问模型

   代码：step6.all_processes_gpt.ipynb



上述代码可以跑通整体流程


# 强化学习训练代码
DPO 训练代码
step4.8multi_image_dpo_train_sampled.py 修改训练集和测试集的路径
```shell
torchrun --nproc-per-node=3 step4.8multi_image_dpo_train_sampled.py
```


----------------------------------------------------------------------------------------------------------------

以下代码针对GPU在国内，连不上chatgpt，所拆分出来的代码



## 1. 使用最新版的prompt获取非top10内的option的结果

### 1.1 先通过pllava生成问题
代码：step4.5Clustering-Copy3.ipynb

运行结果： ./experiment_res/interval_prompt_without_option_newprompt_13b.jsonl

### 1.2 根据上述问题使用全流程的代码获取排序，可以查看经过一轮提问之后，检索效果变化多少
代码：step5.1non_top10_optimization.ipynb中的【获取排序】模块

运行结果：rank_res/non_top10_res_newprompt.jsonl


## 2. 生成DPO 训练数据
### 2.1 根据一个option和5张图片生成5个问题
step4.6GenFirstQuestionWith5Response.ipynb / step4.6GenFirstQuestionWith5Response.py

结果文件路径：./experiment_res/one_option_five_images_five_response_13b.jsonl

### 2.2 非top10的option生成的训练数据
代码：step5.1non_top10_optimization.py

运行结果： ./rank_res/generate_choice_rejected_tmp_nontop10.jsonl

### 2.3 根据上述文件生成DPO需要的数据格式，一个正样本对应一个负样本。
代码：step4.6.1.GenerateChoiceRejectData.ipynb

运行结果： ./playground/data/css_data/dpo_test_dataset.json    和  ./playground/data/css_data/dpo_train_dataset.json


