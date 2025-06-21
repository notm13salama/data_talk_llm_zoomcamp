import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def make_batches(seq, n):
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result

def compute_embeddings(texts, batch_size=8):
    text_batches = make_batches(texts, 8)
    
    all_embeddings = []
    
    for batch in tqdm(text_batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    
        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state
            
            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)
    
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings