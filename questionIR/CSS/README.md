# NavigationIR submitted to NAACL

# Downloads

1. BLIP_FT weight, download to -> ：./BLIP/chatir_weights.ckpt
   download link：https://drive.google.com/file/d/1pW-Rt51yfsocK3Qnyom4hcxA22I7b-nL/view?usp=drive_link

```
.
└── BLIP
    ├── BLIP.gif
    ├── chatir_weights.ckpt

    ... omit other files
```


2. recall weight for Visdial

   download link：https://drive.google.com/drive/folders/1DG6pYotka2RMtcDJcNoRgl6Z2IIGo3sX?usp=share_link

./checkpoints/blip_faiss.index

./checkpoints/id2image.pickle

./checkpoints/blip_image_embedding.pickle

```/checkpoints$ tree
.
└── checkpoints
    ├── blip_faiss.index
    ├── id2image.pickle
    ├── blip_image_embedding.pickle
```



3. PLlava weight -> ./MODELS/

```shell
git clone https://huggingface.co/ermu2001/pllava-7b
```
or https://huggingface.co/ermu2001/pllava-7b/tree/main 
```
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

4. download the pictures -> /playground/data/css_data/unlabeled2017
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


5. for Blip2 download address
https://hf-mirror.com/Salesforce/blip2-opt-2.7b


