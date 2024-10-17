

import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)
import time
import math
import json
import logging
import random
import numpy as np
import shutil
import torch
import torch.nn as nn
from PIL import Image
from accelerate import Accelerator, init_empty_weights, dispatch_model, infer_auto_device_map,load_checkpoint_in_model
from transformers import get_cosine_schedule_with_warmup
from safetensors import safe_open
from pllava import PllavaProcessor, PllavaForConditionalGeneration, PllavaConfig
from accelerate.utils import get_balanced_memory
from argparse import ArgumentParser
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

torch.autograd.set_detect_anomaly(True) 


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)

accelerate = Accelerator()


# Simple version
class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        """
        Args:
            data (list of dict): List of data dictionaries, each containing images and conversations.
            processor: A processor for the model.
        """
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = item['image_paths']  # 5 images per batch
        option = item['option'].replace('<image>', '').strip()
        choose = item['choose']
        reject = item['rejected']

        # Load images
        image_tensor = []
        for image_path in image_paths:
            try:

                image = Image.open( image_path).convert("RGB")
                image_tensor.append(image)
            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")

                placeholder_image = Image.new('RGB', (224, 224), (0, 0, 0))
                image_tensor.append(placeholder_image)

        instruction = f"""You are Pllava, a large vision-language assistant. You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language.
                 Follow the instructions carefully and explain your answers in detail based on the provided video.
                  USER:<image>  USER: You need to find a common object that appears in all 5 pictures but has distinguishing features. Based on this object, ask a question to differentiate the pictures.
                  
                Remember, you must ensure the question is specific, not abstract, and the answer should be directly obtainable by looking at the images.
                
                For example:
                Example 1: All 5 pictures have people, but the number of people differs. You can ask about the number of people.
                Example 2: All 5 pictures have cats, but the colors are different. You can ask about the color.
                Example 3: All 5 pictures have traffic lights, but their positions differ. You can ask about the position of the traffic lights.
                
                Ask a specific question based on the object that will help distinguish the pictures. The question is not overlapped with the description: {option}.
                Don't ask 2 questions each time. such as what is the attribute of a or b.
                
                Output as the following format
                {{
                "Question to differentiate the pictures":""
                }}
                ""
                ASSISTANT:
                """

        choose_prompt = f"""{instruction} {{"Question to differentiate the pictures":"{choose}"}}"""


        choose_encode = self.processor(choose_prompt, image_tensor, return_tensors="pt")

        choose_labels = choose_encode['input_ids'].clone().squeeze(0)
        instruction_input_length = len(
            self.processor.tokenizer(instruction, add_special_tokens=False, padding=True, )['input_ids'])

        choose_labels[:instruction_input_length] = -100  # ignore_index


        reject_prompt = f"""{instruction} {{"Question to differentiate the pictures":"{reject}"}}"""


        reject_encode = self.processor.tokenizer(reject_prompt, padding=True, return_tensors="pt")
        reject_labels = reject_encode['input_ids'].clone().squeeze(0)
        reject_labels[:instruction_input_length] = -100

        return {'input_ids_choose': choose_encode['input_ids'].squeeze(0),  
                'attention_mask_choose': choose_encode['attention_mask'].squeeze(0),  
                'labels_choose': choose_labels,
                'pixel_values': choose_encode['pixel_values'],  
                'input_ids_reject': reject_encode['input_ids'].squeeze(0),
                'attention_mask_reject': reject_encode['attention_mask'].squeeze(0),
                'labels_reject': reject_labels,
                }


def load_pllava(repo_id, num_frames, use_lora=False, weight_dir=None,
                lora_alpha=32, use_multi_gpus=False,
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

        if 'model' in state_dict.keys():
            with torch.device('meta'): 
                msg = model.load_state_dict(state_dict['model'], strict=False, assign=True)
        else:
            with torch.device('meta'):
                msg = model.load_state_dict(state_dict, strict=False, assign=True)

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
        print('Each layer corresponds to the GPU ID', model.hf_device_map)

    return model


def get_prob_log(model, input_ids, attention_mask,pixel_values, labels):

    output = model(input_ids=input_ids,
                          pixel_values=pixel_values,
                   attention_mask=attention_mask,
                          labels = labels,
                          media_type='video')
    shift_logits = output.logits[:, :-1, :].contiguous()  
    shift_labels = output['labels_with_image_token'][:, 1:].contiguous() # assign


    shift_labels = torch.where(shift_labels==-100, torch.tensor(0, device=shift_labels.device), shift_labels)
    loss_mask = (shift_labels != 0)
    per_token_logps = torch.gather(shift_logits.log_softmax(-1), dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    log_probs = (per_token_logps * loss_mask).sum(-1)
    return log_probs


def prepare_dataloader(args):
    processor = PllavaProcessor.from_pretrained(args.repo_id) 
    # processor = PllavaProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf') #
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.eos_token = processor.tokenizer.pad_token


    with open("./playground/data/css_data/dpo_train_dataset.json", 'r') as f:
        data = json.load(f)#[:10]
    train_dataset = MultiModalDataset(data, processor)
    with open("./playground/data/css_data/dpo_test_dataset.json", 'r') as f:

        data = json.load(f)#[:10]
    eval_dataset = MultiModalDataset(data, processor)

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
        input_ids_choose = nn.utils.rnn.pad_sequence([item['input_ids_choose'] for item in batch], batch_first=True,
                                                     padding_value=processor.tokenizer.pad_token_id)
        attention_mask_choose = nn.utils.rnn.pad_sequence([item['attention_mask_choose'] for item in batch],
                                                          batch_first=True, padding_value=0)
        labels_choose = nn.utils.rnn.pad_sequence([item['labels_choose'] for item in batch], batch_first=True,
                                                  padding_value=-100)
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)  # flatten imagesi

        input_ids_reject = nn.utils.rnn.pad_sequence([item['input_ids_reject'] for item in batch], batch_first=True,
                                                     padding_value=processor.tokenizer.pad_token_id)
        attention_mask_reject = nn.utils.rnn.pad_sequence([item['attention_mask_reject'] for item in batch],
                                                          batch_first=True, padding_value=0)
        labels_reject = nn.utils.rnn.pad_sequence([item['labels_reject'] for item in batch], batch_first=True,
                                                  padding_value=-100)

        return {'input_ids_choose': input_ids_choose,  # shape: [batch_size, seq_len]
                'attention_mask_choose': attention_mask_choose,  # shape: [batch_size, seq_len]
                'labels_choose': labels_choose,
                'pixel_values': pixel_values,  # shape: [batch_size * num_images, 3, 224, 224]
                'input_ids_reject': input_ids_reject,
                'attention_mask_reject': attention_mask_reject,
                'labels_reject': labels_reject,
                }

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              collate_fn=collate_fn, shuffle=True,
                                              pin_memory=True, num_workers=4, prefetch_factor=2)
    validloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                              collate_fn=collate_fn, shuffle=False,
                                              pin_memory=True, num_workers=4, prefetch_factor=2)

    return trainloader, validloader

def prepare_model_and_optimizer(args, num_training_steps, num_warmup_steps):
    model = load_pllava(repo_id=args.repo_id,  # 'MODELS/pllava-7b'
                       num_frames=5,  # num_images = 5
                       use_lora=True,
                       weight_dir=args.repo_id, # 'MODELS/pllava-7b'
                       lora_alpha=4,
                       use_multi_gpus=True, #
                       pooling_shape=(5, 12, 12)
                        )
    model.train()
    print("""Forzen the layers of vision_tower and mm_projector""")
    for name, param in model.named_parameters():
        if 'vision_tower' in name or 'projector' in name:  # freeze CLIP
            param.requires_grad = False
    print('trainable parameters', '-'*50, )
    for name, param in model.named_parameters():
        if param.requires_grad:# = False
            print(name, 'requires_grad: ', param.requires_grad)
    # model.print_trainable_parameters()

    # ref model and put the model into gpu
    model_ref = load_pllava(repo_id=args.repo_id,  # 'MODELS/pllava-7b'
                        num_frames=5,  # num_images = 5
                        use_lora=True,
                        weight_dir=args.repo_id,  # 'MODELS/pllava-7b'
                        lora_alpha=4,
                        use_multi_gpus=True,
                        pooling_shape=(5, 12, 12)
                        )
    model_ref.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # Create the learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return model, model_ref, optimizer, scheduler

def evaluate(model, validloader, accelerator: Accelerator, processor):
    model.eval()
    total_loss, total_items = 0, 0
    output_examples, input_examples = [], []
    with torch.inference_mode():
        for batch in validloader:
            # compute loss
            with torch.inference_mode():
                # compute loss
                output = model(input_ids=batch['input_ids_choose'],
                               attention_mask=batch['attention_mask_choose'],
                               labels=batch['labels_choose'],
                               pixel_values=batch['pixel_values'],
                               media_type='video')
                total_loss += output.loss.item() * batch['input_ids_choose'].size(0)
                total_items += batch['input_ids_choose'].size(0)

    average_loss = total_loss / total_items
    accelerator.print('some example during evaluating')
    model.train()
    return average_loss

def train(model, model_ref, optimizer, scheduler, trainloader, validloader,
          accelerator: Accelerator, resume, epoch=3, log_step=100,
          save_interval=5, model_tokenizer=None):
    global_step = 0
    start_time = time.time()

    resume_step = 0
    resume_epoch = 0
    checkpoint_history = []

    if resume is not None:
        accelerator.load_state(resume)
        steps_per_epoch = math.ceil(len(trainloader) / accelerator.gradient_accumulation_steps)
        resume_step = global_step = int(resume.split("step_")[-1])
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch 
        accelerator.print(f"resume from checkpoint -> {resume}")

    for ep in tqdm(range(resume_epoch, epoch)):
        model.train()
        model_ref.eval()
        if resume and ep == resume_epoch and resume_step != 0: 
            active_dataloader = accelerator.skip_first_batches(
                trainloader,
               resume_step * accelerator.gradient_accumulation_steps 
              )
        else:
            active_dataloader = trainloader
        for batch in tqdm(active_dataloader):
            with accelerator.accumulate(model): # gradient accumulate
                optimizer.zero_grad()
                # good
                model_logps_choose = get_prob_log(model=model, input_ids=batch['input_ids_choose'],
                                                  attention_mask=batch['attention_mask_choose'],
                                           pixel_values=batch['pixel_values'],
                                           labels=batch['labels_choose'])
                # bad
                model_logps_reject = get_prob_log(model=model, input_ids=batch['input_ids_reject'],
                                                  attention_mask=batch['attention_mask_reject'],
                                                  pixel_values=batch['pixel_values'],
                                                  labels=batch['labels_reject'])
                with torch.no_grad():
                    modelref_logps_choose = get_prob_log(model=model_ref,
                                                         input_ids=batch['input_ids_choose'],
                                                         attention_mask=batch['attention_mask_choose'],
                                                         pixel_values=batch['pixel_values'],
                                                         labels=batch['labels_choose'])
                    modelred_logps_reject = get_prob_log(model=model_ref, input_ids=batch['input_ids_reject'],
                                                         attention_mask=batch['attention_mask_reject'],
                                                  pixel_values=batch['pixel_values'],
                                                  labels=batch['labels_reject'])
                # compute loss
                reward_choose = (model_logps_choose - modelref_logps_choose).mean()
                reward_reject = (model_logps_reject - modelred_logps_reject).mean()
                loss = -1 * nn.functional.logsigmoid(0.1 * (reward_choose - reward_reject)).mean()
                # reduce loss from multi-gpus
                reward_reject = accelerator.reduce(reward_reject, 'mean')
                reward_choose = accelerator.reduce(reward_choose, 'mean')
                loss = accelerator.reduce(loss, "mean")
                accelerator.backward(loss, retain_graph=True)
                optimizer.step()
                scheduler.step() # update learning rate

                if accelerator.sync_gradients: 
                    global_step += 1
                    accelerator.log({"train_loss": loss.item(), 'reward_choose': reward_choose.item(),
                                     'reward_reject': reward_reject.item(),
                                     'learning_rate':scheduler.get_last_lr()[0]}, global_step)

                    if global_step % log_step == 0:
                        accelerator.print(f"epoch: {ep}, global_step: {global_step}, loss: {loss.item()}, "
                                          f"reward_choose: {reward_choose.item()}, reward_reject:{reward_reject.item()}, "
                                          f"LR: {scheduler.get_last_lr()[0]}")

                    if global_step % save_interval == 0 and global_step != 0:
                        accelerator.print(f"save checkpoint -> step_{global_step}")
                        accelerator.save_state(accelerator.project_dir + f"/step_{global_step}")

                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/step_{global_step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save
                        )

                        checkpoint_history.append(accelerator.project_dir + f"/step_{global_step}")
                        if len(checkpoint_history) > 3: 
                            oldest_checkpoint = checkpoint_history.pop(0)
                            shutil.rmtree(oldest_checkpoint)
                            accelerator.print(f"Deleted old checkpoint -> {oldest_checkpoint}")
                    eval_loss = evaluate(model, validloader, accelerator, processor=model_tokenizer)
                    accelerator.print(f"epoch: {ep}, eval_loss: {eval_loss}, time: {time.time() - start_time}")
                    accelerator.log({"eval_loss(CrossEntropy)": eval_loss}, global_step)

    accelerator.end_training() # Tensorborad

def parse_args():
    parser = ArgumentParser(description="Training script for multi-image pretraining.")
    parser.add_argument('--repo_id', type=str, default='MODELS/pllava-7b', help="Model weight from pretrained")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--log_step", type=int, default=1, help="Frequency of logging training information")
    parser.add_argument("--save_interval", type=int, default=17, help="Interval of saving checkpoints")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of gradient accumulation steps")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,

                              log_with="tensorboard", project_dir="ckpts")
    accelerator.init_trackers("runs")  # tensorboard
    trainloader, validloader = prepare_dataloader(args)

    num_training_steps = len(trainloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)  

    model, model_ref, optimizer, scheduler = prepare_model_and_optimizer(args, num_training_steps, num_warmup_steps)
    
    optimizer, trainloader, validloader = accelerator.prepare(optimizer, trainloader, validloader)

    train(model=model, model_ref=model_ref, optimizer=optimizer, scheduler=scheduler, trainloader=trainloader,
          validloader=validloader, accelerator=accelerator, resume='./ckpts/step_51',
          epoch=args.epochs, log_step=args.log_step, save_interval=args.save_interval,
          model_tokenizer=trainloader.dataset.processor)

if __name__ == "__main__":
    main()

