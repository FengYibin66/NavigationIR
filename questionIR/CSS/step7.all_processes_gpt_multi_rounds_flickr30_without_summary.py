# %% [markdown]


# %%
import pandas as pd
import faiss 
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image 
import pickle 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import random
import os
from openai import OpenAI

import base64
import requests

import json
import re 



# %%

def show_image(image_paths, sentence=None):


    k=(len(image_paths)+4)//5 
    fig, axs = plt.subplots(nrows=k, ncols=5, figsize=(20, 8))  
    axs = axs.flatten()  

    
    for ax, img_path in zip(axs, image_paths):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')  
            ax.set_title(img_path.split('/')[-1])  
        except FileNotFoundError:
            ax.imshow(np.zeros((10, 10, 3), dtype=int))  
            ax.axis('off')
            ax.set_title('File Not Found')
    if sentence:
        fig.suptitle(sentence, fontsize=16)
    plt.tight_layout()
    plt.show()



class ImageEmbedder:
    def __init__(self, model, preprocessor):
        """ model projects image to vector, processor load and prepare image to the model"""
        self.model = model
        self.processor = preprocessor

def BLIP_BASELINE():
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    import sys
    sys.path.insert(0, './BLIP')
    from BLIP.models.blip_itm import blip_itm
    # load model
    model = blip_itm(pretrained='./BLIP/chatir_weights.ckpt',  # Download from Google Drive, see README.md
                     med_config='BLIP/configs/med_config.json',
                     image_size=224,
                     vit='base'
                     )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # define Image Embedder (raw_image --> img_feature)
    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    def blip_project_img(image):
        embeds = model.visual_encoder(image)
        projection = model.vision_proj(embeds[:, 0, :])
        return F.normalize(projection, dim=-1)

    def blip_prep_image(path):
        raw = Image.open(path).convert('RGB')
        return transform_test(raw)

    image_embedder = ImageEmbedder(blip_project_img, lambda path: blip_prep_image(path))

    # define dialog encoder (dialog --> img_feature)
    def dialog_encoder(dialog):
        text = model.tokenizer(dialog, padding='longest', truncation=True,
                               max_length=200,
                               return_tensors="pt"
                               ).to(device)

        text_output = model.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                         return_dict=True, mode='text')

        shift = model.text_proj(text_output.last_hidden_state[:, 0, :])
        return F.normalize(shift, dim=-1)

    return dialog_encoder, image_embedder

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list):
        """
        Args:
            texts (list of str): List of text strings.
            processor (transformers processor): Processor to tokenize the text.
        """
        self.texts = texts
        # self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]  # Get the text at the provided index
        return {'text': text}

def encode_text(dataset, model):
    """CLIP for encode text """
    # model.eval()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=20,
                                              pin_memory=True,
                                              num_workers=1,
                                              prefetch_factor=2,
                                              shuffle=False,
                                              
                                              )
    all_features = []
    with torch.no_grad():
        for batch in data_loader:
            features = model(batch['text'])
            all_features.append(features.cpu())  
    return torch.cat(all_features)  



def retrieve_topk_images(query: list,
                         topk=10,
                         faiss_model=None,
                         blip_model=None,
                         id2image=None,
                         processor=None, ):
    text_dataset = TextDataset(query)
    query_vec = encode_text(text_dataset, blip_model)
    query_vec = query_vec.numpy()
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    

    distance, indices = faiss_model.search(query_vec, topk)
    
    indices = np.array(indices)
    image_paths = [[id2image.get(idx, 'path/not/found') for idx in row] for row in indices]
    
    
    return image_paths, indices



def find_index_in_list(element, my_list):
    return my_list.index(element) 






# %%
import json
class ImageDataset(torch.utils.data.Dataset):
    """ Dataset class for the corpus images (the 50k potential candidates)"""
    def __init__(self, image_paths, preprocessor):
        self.image_paths = image_paths
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.preprocessor(image_path)  
        return {'id': idx, 'image': image}

# %%
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def encode_images(dataset, image_embedder):
    # model.eval()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=500,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True,
                                             drop_last=False,
                                             prefetch_factor=2
                                             )
    print("Preparing corpus (search space)...")
    corpus_vectors = []
    # corpus_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_vectors = F.normalize(image_embedder.model(batch['image'].to(device)), dim=-1)
            corpus_vectors.append(batch_vectors)
            # corpus_ids.append(batch['id'].to(device))

        corpus_vectors = torch.cat(corpus_vectors)
    return corpus_vectors

# %%

dialog_encoder, image_embedder = BLIP_BASELINE()

image_fold = './playground/data/flickr30k-images'
image_paths = os.listdir(image_fold)#[:20]
merge_image_path = [os.path.join(image_fold, p) for p in image_paths]
merge_image_path = sorted(merge_image_path)
id2image = dict(zip(range(len(merge_image_path)), merge_image_path))
with open('./checkpoints/id2image_flickr30k_BLIP_FT.pickle', 'wb') as f:
    pickle.dump(id2image, f)

dataset = ImageDataset(merge_image_path, image_embedder.processor)

encoded_images = encode_images(dataset, image_embedder)

# %%
encoded_images.shape

# %%
encoded_images = encoded_images.cpu().numpy()
encoded_images /= np.linalg.norm(encoded_images, axis=1, keepdims=True) 


dimension = encoded_images.shape[1]
faiss_index = faiss.IndexFlatIP(dimension) # 
faiss_index.add(encoded_images)
faiss.write_index(faiss_index,'./checkpoints/blip_faiss_flickr30k_BLIP_FT.index')

# %%
with open('checkpoints/blip_image_embedding_flickr30k_BLIP_FT.pickle', 'wb') as f:
    pickle.dump(encoded_images, f)

# %%


def generate_valid_question(base64_image_top, system_prompt, previous_questions, max_retries=5):
    
    for attempt in range(max_retries):
        try:
            

            
            question_fewshot = question_attribute(base64_image_top, system_prompt, previous_questions)

            
            question_fewshot_json = extract_json(question_fewshot)

            if question_fewshot_json:
                question_fewshot_json = json.loads(question_fewshot_json)  
                if "Question to differentiate the pictures" in question_fewshot_json:
                    return question_fewshot_json["Question to differentiate the pictures"]
                else:
                    print(f"Key 'Question to differentiate the pictures' not found, retrying...")

        except json.JSONDecodeError:
            print("JSONDecodeError: The result is not valid JSON, retrying...")
        except KeyError:
            print("KeyError: Expected key not found in the generated JSON, retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}, retrying...")
        
        
        attempt += 1


    print("Failed to generate a valid question after multiple attempts.")
    return None


# %%
def extract_json(text):
    pattern = r'{.*}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group()
    else:
        return 'parse incorrectly'

# %%
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def answer(question,base64_target_image):
    client = OpenAI(api_key=,base_url ='') 
    
    response = client.chat.completions.create(
      model="",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"according to the image, answer the question:{question}，Your answer must be direct and simple",
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_target_image}",
              },
            }
          ],
        }
      ],
      max_tokens=100,
    )
    return response.choices[0].message.content

# %%
system_prompt = """
You are an assistant that helps users to differentiate between images by generating specific, non-repetitive questions based on given images. 
Given 5 pictures and a list of questions you have previously asked, your task is to find common objects that appear in all 5 pictures but have distinguishing features. Based on these objects, ask a question that is not included in [previous questions] to differentiate the pictures.
Make sure the question is specific, directly related to the images, and not abstract. Avoid repeating the same attributes mentioned in any previous questions. And don't ask two questions at the same time, such as 'What is the attribute of a or b?'.


For example:
Example 1: All 5 pictures have people, but the number of people differs. You can ask about the number of people.
Example 2: All 5 pictures have cats, but the colors are different. You can ask about the color.
Example 3: All 5 pictures have traffic lights, but their positions differ. You can ask about the position of the traffic lights.


ensure that the output strictly(have to !) follows this JSON format:
{
"What is the common object that appears in all five pictures":"Answer1",
"What is the distinguishing feature that can help differentiate the picture":"Answer2",
"Question to differentiate the pictures":"Answer3"
}
The output should not contain any extra characters or text outside of this JSON structure.

"""

# %%
def question_attribute(base64_image_list, system_prompt, previous_questions):
    client = OpenAI(api_key=, base_url="")  

    
    previous_questions_text = " ".join([f'Previous Question: {q}' for q in previous_questions])
    
    
    user_prompt = f"""
    [Previous questions]: {previous_questions_text}.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},  
        {"role": "user","content": [{"type": "text", "text": user_prompt}]},  
      ]
        
    for img_base64 in base64_image_list:
        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            }]
        })
    
    response = client.chat.completions.create(
      model="",
      messages=messages,
      max_tokens=500,
    )
    
    return response.choices[0].message.content


# %%

def summary(query,question,answer):
    client = OpenAI(api_key=,base_url ='') 
    
    response = client.chat.completions.create(
    model="",
    messages=[
        {"role": "system", "content": f"""
        Your task is to summarize the information from the image's question and answer and add this information to the original image description.\
        Remember: the summarized information must be concise, and the original description should not be altered.

        <question>
        {question}
        <answer>
        {answer}
        <image description>
        {query}


The information extracted from the question and answer should be added to the original description as an attribute or a simple attributive clause.
        """
        },
        {"role": "user", "content": ""}
      ]
    )
    return response.choices[0].message.content

# %%
import faiss
import pickle

try:
    print("Load faiss Model...")
    faiss_model = faiss.read_index('./checkpoints/blip_faiss_flickr30k_BLIP_FT.index')
    print("faiss faiss Model Load successful")
except Exception as e:
    print(f"Error: {e}")

try:
    print("Load id2image Data...")
    with open('./checkpoints/id2image_flickr30k_BLIP_FT.pickle', 'rb') as f:
        id2image = pickle.load(f)
    print("id2image DataLoad Successful")
except Exception as e:
    print(f"Load id2image Error: {e}")

try:
    print("Load image_vector Data...")
    with open('./checkpoints/blip_image_embedding_flickr30k_BLIP_FT.pickle', 'rb') as f:
        image_vector = pickle.load(f)
    print("image_vector DataLoad Successful")
except Exception as e:
    print(f"Load image_vector Error: {e}")


# %%
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/.autodl/HYF/questionIR/CSS/MODELS/bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('/root/autodl-tmp/.autodl/HYF/questionIR/CSS/MODELS/bert-base-uncased') 




# %%

faiss_model = faiss.read_index('./checkpoints/blip_faiss_flickr30k_BLIP_FT.index')
with open('./checkpoints/id2image_flickr30k_BLIP_FT.pickle', 'rb') as f:
    id2image = pickle.load(f)
    
with open('./checkpoints/blip_image_embedding_flickr30k_BLIP_FT.pickle', 'rb') as f:
    image_vector = pickle.load(f)
dialog_encoder, image_embedder = BLIP_BASELINE()


class DataLoader:
    def load_data(self):
        raise NotImplementedError("This method should be covered by subclass")

# %%

class CSVDataLoader(DataLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)

# %%
from datasets import load_dataset

class HuggingFaceDataLoader(DataLoader):
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split

    def load_data(self):
        dataset = load_dataset(self.dataset_name, split=self.split)
        return dataset.to_pandas()  

# %%

def get_data_loader(file_type, file_path=None, dataset_name=None, split=None):
    if file_type == "csv":
        return CSVDataLoader(file_path)
    elif file_type == "huggingface":
        return HuggingFaceDataLoader(dataset_name, split)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# %%


# %%
# csv_data.head()



# %%

def run_sample_case(file_type, file_path=None, dataset_name=None, split=None):
    
    data_loader = get_data_loader(file_type, file_path=file_path, dataset_name=dataset_name, split=split)
    data = data_loader.load_data()  # LoadData

    print(data.shape)  
    
    
    i = 0
    
    print(data.iloc[i])
    
    raw_str = data.iloc[i]["raw"]
    raw_list = json.loads(raw_str)  
    query = raw_list[0]  
    print(f'query: {query}')
    target_image_path = './playground/data/flickr30k-images/' + data.iloc[i]["filename"]  
    
    
    image_paths, indices = retrieve_topk_images([query], topk=50000, faiss_model=faiss_model, blip_model=dialog_encoder, id2image=id2image, processor=None)

    image_rank = find_index_in_list(target_image_path, image_paths[0])

    
    
    initial_rank = image_rank
    max_rounds = 5
    rank_threshold = 10
    round_number = 0
    accumulated_summary = ""
    previous_questions = [query]
    image_ranks_per_round = []
    
    
    query_with_feedback = "Initial Query: " + query + accumulated_summary  


    
    while round_number < max_rounds:
        round_number += 1
        print(f"Round {round_number}:")
        
        
        top_images_path = image_paths[0][:40]
        random_selection_path = random.sample(top_images_path, 5)
        base64_image_top = [encode_image(image_path) for image_path in random_selection_path]
        
        
        question = generate_valid_question(base64_image_top, system_prompt, previous_questions)
        previous_questions.append(question)
        
        
        base64_target_image = encode_image(target_image_path)
        answer_of_question = answer(question, base64_target_image)
        
        print(f'new -> question: {question}, answer_of_question: {answer_of_question}')
        
        
        
        accumulated_summary = f" Question: {question} Answer: {answer_of_question}"
        
        
        query_with_feedback += accumulated_summary  
        print(f'query_with_feedback: {query_with_feedback}, ')
        
        
        # summary_of_question_and_option = summary(query_with_feedback, question, answer_of_question)
        
        # 直接使用 query_with_feedback 进行检索，不生成 summary_of_question_and_option
        image_paths_new, indices = retrieve_topk_images([query_with_feedback], topk=50000, faiss_model=faiss_model, blip_model=dialog_encoder, id2image=id2image, processor=None)
        
        
        image_rank_new = find_index_in_list(target_image_path, image_paths_new[0])
        image_ranks_per_round.append((image_rank_new, image_paths_new[0]))  
        
        
        print(f"Old Rank: {image_rank}, New Rank: {image_rank_new}")
        
        
        if image_rank_new < rank_threshold:
            print(f"Image rank ({image_rank_new}) is below the threshold ({rank_threshold}). Ending early.")

            break
        
        
        image_paths = image_paths_new
        image_rank = image_rank_new

    
    best_rank, best_image_paths = min(image_ranks_per_round, key=lambda x: x[0])  
    best_image_path = best_image_paths[0]  

    
    if best_rank < initial_rank:
        print(f"Final Best Rank improved from {initial_rank} to {best_rank}, Best Image Path: {best_image_path}")
    else:
        print(f"No significant improvement. Final Best Rank: {best_rank}, Best Image Path: {best_image_path}")
        
    show_image(image_paths[0][:best_rank+1], sentence=None)





run_sample_case(file_type="csv", file_path="/root/autodl-tmp/.autodl/HYF/questionIR/CSS/downloads/flickr30k/flickr_annotations_filtered_30k.csv")





folder_path = '/root/autodl-tmp/.autodl/HYF/questionIR/CSS/downloads/flickr30k/'
file_name_list = [
    'flickr_annotations_filtered_30k.csv',
]


def init_csv_file(output_file):
    fieldnames = [
        'file_name', 
        'query_number', 
        'initial_rank', 
        'best_rank', 
        'rounds', 
        'image_ranks_per_round', 
        'top_10_hits_before', 
        'top_10_hits_after',
        'queries_feedback',  
        'average_rank'
    ]
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return fieldnames



def save_results_to_csv(output_file, fieldnames, file_name, query_number, initial_rank, best_rank, rounds, image_ranks_per_round, top_10_hits_before, top_10_hits_after, queries_feedback):
    

    ranks = list(map(int, image_ranks_per_round.split(',')))  
    average_rank = sum(ranks) / len(ranks) if ranks else 0
    
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'file_name': file_name,
            'query_number': query_number,
            'initial_rank': initial_rank,
            'best_rank': best_rank,
            'rounds': rounds,
            'image_ranks_per_round': image_ranks_per_round,
            'top_10_hits_before': top_10_hits_before,
            'top_10_hits_after': top_10_hits_after,
            'queries_feedback': queries_feedback,  
            'average_rank': average_rank  
        })



def read_csv(file_path):

    data = pd.read_csv(file_path)
    return data

# %%
import csv
from tqdm import tqdm

query_question = []


max_rounds = 5
rank_threshold = 10  


top_10_hits_before = 0  
top_10_hits_after = 0  
total_queries_accumulated = 0  


output_file = 'multi-round-result_Blip_FT_flickr30k.csv'


fieldnames = init_csv_file(output_file)



for file_name in file_name_list:
    file_path = folder_path + file_name
    csv_data = read_csv(file_path)  

    total_queries = len(csv_data)  
    total_queries_accumulated += total_queries  
    

    top_10_hits_before_dataset = 0  
    top_10_hits_after_dataset = 0  
    



    for i in tqdm(range(total_queries)):

        raw_str = csv_data.iloc[i]["raw"]
        raw_list = json.loads(raw_str)  
        query = raw_list[0]  
        
        target_image_path = './playground/data/flickr30k-images/' + csv_data["filename"][i]  
        

        image_ranks_per_round = []
        queries_feedback = []  
        

        image_paths, indices = retrieve_topk_images([query],
                                                    topk=50000,
                                                    faiss_model=faiss_model,
                                                    blip_model=dialog_encoder,
                                                    id2image=id2image,
                                                    processor=None)
        image_rank = find_index_in_list(target_image_path, image_paths[0])
        

        if image_rank < 10:
            top_10_hits_before += 1
            top_10_hits_before_dataset += 1


        initial_rank = image_rank
        

        accumulated_summary = ""
        previous_questions = [query]  
        query_with_feedback = "Initial Query: " + query + accumulated_summary  
        
        round_number = 0

        while round_number < max_rounds:
            round_number += 1

            top_images_path = image_paths[0][:40]
            random_selection_path = random.sample(top_images_path, 5)
            base64_image_top = [encode_image(image_path) for image_path in random_selection_path]
            

            question = generate_valid_question(base64_image_top, system_prompt, previous_questions)

            if question is None:
                print(f"Failed to generate a valid question for image {i+1}, skipping this sample.")
                continue
            
            query_question.append([query_with_feedback, question])
            previous_questions.append(question)  
            

            base64_target_image = encode_image(target_image_path)
            answer_of_question = answer(question, base64_target_image)
            

            accumulated_summary = f" Question: {question} Answer: {answer_of_question}"
            
            query_with_feedback += accumulated_summary  

            # summary_of_question_and_option = summary(query_with_feedback, question, answer_of_question)
            
            
            image_paths_new, indices = retrieve_topk_images([query_with_feedback],
                                                            topk=50000,
                                                            faiss_model=faiss_model,
                                                            blip_model=dialog_encoder,
                                                            id2image=id2image,
                                                            processor=None)
            

            image_rank_new = find_index_in_list(target_image_path, image_paths_new[0])
            image_ranks_per_round.append(image_rank_new)  

            if image_rank_new < rank_threshold:
                print(f"Image rank ({image_rank_new}) is below the threshold ({rank_threshold}). Ending early.")
                break  
            

            image_paths = image_paths_new

        queries_feedback.append({
            'query_with_feedback': query_with_feedback,
            # 'question': question,
            # 'answer': answer_of_question
        }) 

        best_rank = min(image_ranks_per_round)  

        if best_rank < 10:
            top_10_hits_after += 1
            top_10_hits_after_dataset += 1

        save_results_to_csv(
            output_file, 
            fieldnames, 
            file_name, 
            i + 1, 
            initial_rank, 
            best_rank, 
            round_number , 
            ','.join(map(str, image_ranks_per_round)), 
            top_10_hits_before_dataset, 
            top_10_hits_after_dataset,
            queries_feedback  
        )
        
        
    
    top_10_recall_rate_before_dataset = top_10_hits_before_dataset / total_queries
    top_10_recall_rate_after_dataset = top_10_hits_after_dataset / total_queries
    print(f'Dataset: {file_name} - Top-10 hits before multi-round: {top_10_hits_before_dataset}, total_queries: {total_queries}')
    print(f"Dataset: {file_name} - Top-10 Recall Rate before multi-round: {top_10_recall_rate_before_dataset * 100:.4f}%")
    print(f'Dataset: {file_name} - Top-10 hits after multi-round: {top_10_hits_after_dataset}, total_queries: {total_queries}')
    print(f"Dataset: {file_name} - Top-10 Recall Rate after multi-round: {top_10_recall_rate_after_dataset * 100:.4f}%")
    print('---------------------------------------')
    

top_10_recall_rate_before = top_10_hits_before / total_queries_accumulated
top_10_recall_rate_after = top_10_hits_after / total_queries_accumulated
print(f'top_10_hits before multi-round: {top_10_hits_before}, total_queries_accumulated: {total_queries_accumulated}')
print(f"Top-10 Recall Rate before multi-round: {top_10_recall_rate_before * 100:.4f}%")
print(f'top_10_hits after multi-round: {top_10_hits_after}, total_queries_accumulated: {total_queries_accumulated}')
print(f"Top-10 Recall Rate after multi-round: {top_10_recall_rate_after * 100:.4f}%")





