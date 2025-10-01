import clip
import torch
import numpy as np
import faiss
from tqdm import tqdm
import json

batch_size = 1
all_embeddings = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
words = []
with open("data", "r") as f:
    for i in f.readlines():
        words.append(i.strip())

for i in tqdm(range(0, len(words), batch_size)):
    batch = words[i:i+batch_size]
    tokens = clip.tokenize(batch).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    all_embeddings.append(emb.cpu().numpy())

word_embeddings = np.vstack(all_embeddings)
dim = word_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(word_embeddings)

# Save FAISS index
faiss.write_index(index, "words.index")

# Build dict: {id: word}
word_map = {i: word for i, word in enumerate(words)}

with open("words.json", "w") as f:
    json.dump(word_map, f)