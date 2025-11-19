from transformers import AutoTokenizer, AutoModel
import torch


# 1. Extract Token-Level Embeddings
model_name = "microsoft/phi-4-multimodal"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

text = "The kitten climbed the tree."

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=False, return_dict=True)

# (batch, seq_len, hidden_dim)
token_embeddings = outputs.last_hidden_state

print(token_embeddings.shape)

# This gives something like: torch.Size([1, seq_len, 3072]) - depending on model version
# These are the vectors you use for token similarities (cosine distance to “cat”, etc.)


# 2. Extract a Single Embedding for the Whole Sentence

# sentence_embedding = token_embeddings.mean(dim=1)  # (batch, hidden_dim)

# Alternatively use first token if the model uses a BOS embedding.

# Optional: Extract the embedding for “cat” (reference vector)

cat_inputs = tokenizer("cat", return_tensors="pt")

with torch.no_grad():
    cat_vec = model(**cat_inputs).last_hidden_state[0, 1]

# Explanation:
# position 1 skips the BOS token (<s> or similar)
# you take the contextual embedding of the word “cat”


# 3. Compute similarity between Phi-4-MM embeddings

import torch.nn.functional as F

# token_embeddings: (1, seq_len, hidden)
# cat_vec: (hidden)

cat_vec_norm = cat_vec.unsqueeze(0).unsqueeze(0)   # shape → (1, 1, hidden)

cos_sim = F.cosine_similarity(token_embeddings, cat_vec_norm, dim=-1)

# map from [-1,1] to [0,1]
cat_score = (cos_sim + 1) / 2

print(cat_score)
# This gives you a per-token cat-relatedness score in [0, 1].


# 4. Extract Embeddings Conditioned on Images (multimodal)
# If you want Phi-4-MM to use the image context when computing text embeddings:

from PIL import Image

image = Image.open("cat.png")
inputs = tokenizer(text, images=image, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)

token_embeddings = out.last_hidden_state
# Important notes for Phi-4-MM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Do not use model.generate() when extracting embeddings — generation uses the LM head, not hidden states.
# The LM head (vocab projection) is irrelevant for embedding similarity; skip it.
# Phi-4-MM uses a unified backbone, so hidden states already contain cross-modal fused information when you pass an image.
# For embedding similarity, you always want: outputs.last_hidden_state, not outputs.logits.