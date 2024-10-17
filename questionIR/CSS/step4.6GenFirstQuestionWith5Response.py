


import pandas as pd
import faiss
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import random, requests, io
import json, re

os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.cuda.set_device(1)

from accelerate import Accelerator

accelerator = Accelerator()
print(accelerator.device)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)



from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from safetensors import safe_open
from pllava import PllavaProcessor, PllavaForConditionalGeneration, PllavaConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_in_model
from accelerate.utils import get_balanced_memory


def load_pllava(repo_id, num_frames, use_lora=False, weight_dir=None, lora_alpha=32, use_multi_gpus=False,
                pooling_shape=(16, 12, 12)):
    kwargs = {
        'num_frames': num_frames,
    }

    if num_frames == 0:
        kwargs.update(pooling_shape=(0, 12, 12))  
    config = PllavaConfig.from_pretrained(
        repo_id if not use_lora else weight_dir,
        pooling_shape=pooling_shape,
        **kwargs,
    )


    model = PllavaForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)

    try:
        processor = PllavaProcessor.from_pretrained(repo_id)
    except Exception as e:
        processor = PllavaProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

    processor.padding_side = 'left'
    processor.tokenizer.padding_side = 'left'

    if use_lora and weight_dir is not None:
        print("Use lora")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, target_modules=["q_proj", "v_proj"],
            r=128, lora_alpha=lora_alpha, lora_dropout=0.
        )
        print("Lora Scaling:", lora_alpha / 128)
        model.language_model = get_peft_model(model.language_model, peft_config)
        assert weight_dir is not None, "pass a folder to your lora weight"
        print("Finish use lora")

    if weight_dir is not None:
        state_dict = {}
        save_fnames = os.listdir(weight_dir)
        if "model.safetensors" in save_fnames:
            use_full = False
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    use_full = True
                    break
        else:
            use_full = True

        if not use_full:
            print("Loading weight from", weight_dir, "model.safetensors")
            with safe_open(f"{weight_dir}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            print("Loading weight from", weight_dir)
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    with safe_open(f"{weight_dir}/{fn}", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)
        with torch.device('meta'):
            if 'model' in state_dict.keys():
                msg = model.load_state_dict(state_dict['model'], strict=False,assign=True)
            else:
                msg = model.load_state_dict(state_dict, strict=False,assign=True)
        print(msg)

    if use_multi_gpus:
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16',
            low_zero=False,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16'
        )

        dispatch_model(model, device_map=device_map)
        print(model.hf_device_map)
    model = model.eval()
    return model, processor

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def extract_json(text):
    pattern = r'{.*}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group()
    else:
        return 'parse incorrectly'

def extract_llm_ouptut(text):
    ext_json = extract_json(text)
    try:
        question = json.loads(ext_json)['Question to differentiate the pictures']
    except:
        question = ''
        pass
    return question


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, option, images_path, processor):
        """
        Args:
            data (list of dict): List of data dictionaries, each containing images and conversations.
            processor: A processor for the model.
        """
        self.option = option
        self.images_path = images_path
        self.processor = processor

    def __len__(self):
        return len(self.option)

    def __getitem__(self, idx):
        image_paths = self.images_path[idx]  
        option = self.option[idx]

        image_tensor = [Image.open(image_path).convert("RGB") for image_path in image_paths]


        prompt = f"""You are Pllava, a large vision-language assistant. 
    # You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language.
    # Follow the instructions carefully and explain your answers in detail based on the provided video.
    #  USER:<image>  USER: You need to find a common object that appears in all 5 pictures but has distinguishing features. Based on this object, ask a question to differentiate the pictures.

                Remember, you must ensure the question is specific, not abstract, and the answer should be directly obtainable by looking at the images.

                For example:
                Example 1: All 5 pictures have people, but the number of people differs. You can ask about the number of people.
                Example 2: All 5 pictures have cats, but the colors are different. You can ask about the color.
                Example 3: All 5 pictures have traffic lights, but their positions differ. You can ask about the position of the traffic lights.

                Ask a specific question based on the object that will help distinguish the pictures. The question is not overlapped with the description: {option}.
                Don't ask 2 questions each time. such as what is the attribute of a or b

                Output as the following format
                {{
                "What is the common object that appears in all five pictures":"",
                "What is he distinguishing feature that can help differentiate the picture":"",
                "Question to differentiate the pictures":""
                }}
                ""
                ASSISTANT:
                """



        encode = self.processor(prompt, image_tensor, return_tensors="pt")

        return {'input_ids': encode['input_ids'].squeeze(0),  
                'attention_mask': encode['attention_mask'].squeeze(0),  
                'pixel_values': encode['pixel_values'],  
                }


model, processor = load_pllava(repo_id='MODELS/pllava-13b', #'llava-hf/llava-1.5-7b-hf',
            num_frames=5, # num_images = 5
            use_lora=True,
            weight_dir='MODELS/pllava-13b',
            lora_alpha=4,
            use_multi_gpus=False,
            pooling_shape=(5,12,12),
            )

def collate_fn(batch):
    """
    Custom collate function to handle the merging of text and image data.
    Args:
        batch: A list of tuples (input_ids, pixel_values) from the dataset.
    Returns:
        A tuple (input_ids, pixel_values) where:
            input_ids: Tensor of shape [batch_size, seq_len]
            pixel_values: Tensor of shape [batch_size * num_images, 3, 224, 224]
    """

    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch],
                                                batch_first=True,
                                                padding_value=processor.tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True,
                                                     padding_value=0)
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)  

    return {'input_ids': input_ids,  
            'attention_mask': attention_mask,  
            'pixel_values': pixel_values,  
            }


data_repeat_5times = pd.read_csv('./experiment_res/data_repeat_5times.csv')

option = data_repeat_5times['option'].tolist()
images_path = data_repeat_5times['interval_sample_images'].tolist()
images_path = [eval(p) for p in images_path]
dataset = MultiModalDataset(option, images_path, processor)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=2,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=4,
                                         collate_fn=collate_fn,
                                         prefetch_factor=2
                                         )
model, dataloader = accelerator.prepare(model, dataloader)

output_res = []
for batch in tqdm(dataloader):
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}

    with torch.no_grad():
        output_token = model.module.generate(**batch, media_type='video',
                                      do_sample=True,
                                      max_new_tokens=1000,
                                      num_beams=1,
                                      min_length=1,
                                      top_p=0.9,
                                      repetition_penalty=1,
                                      length_penalty=1,
                                      temperature=0.9,
                                      )  
    output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    input_text = processor.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for inp, out in zip(input_text, output_text):
        llm_out = out.split('ASSISTANT:')[-1].strip()

        processed_text = extract_llm_ouptut(str(llm_out))
        output_res.append(processed_text)


        save_dict = {
            'option': inp,
            'original_output': llm_out,
            'questions': processed_text
        }
        with open('./experiment_res/one_option_five_images_five_response_13b.jsonl', 'a') as f:
            json_str = json.dumps(save_dict)  
            f.write(json_str + '\n')
