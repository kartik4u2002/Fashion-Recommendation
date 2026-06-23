import pandas as pd
import numpy as np
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import pillow_avif

# 1. Color extraction
def extract_color(row):
    p_id = row['id']
    if p_id == 'myntra_29132512':
        return 'Lavender'
    elif p_id == 'nykaa_22006528':
        return 'Black'
    desc = str(row['description']).lower()
    name = str(row['name']).lower()
    text = name + " " + desc
    colors = ['navy blue', 'navy', 'off white', 'sea green', 'dark grey', 'light grey', 'white', 'black', 'grey', 'gray', 'beige', 'blue', 'red', 'green', 'yellow', 'pink', 'purple', 'orange', 'brown', 'gold', 'silver', 'olive', 'maroon', 'peach', 'cream', 'teal', 'khaki', 'rust', 'mustard', 'tan', 'wine', 'turquoise', 'lilac', 'lavender', 'rose']
    for c in colors:
        if c in text:
            val = c.title()
            if val == 'Gray':
                val = 'Grey'
            if val == 'Navy':
                val = 'Navy Blue'
            return val
    return 'Unknown'

print("Loading products.csv...")
df = pd.read_csv('products.csv')
df['color'] = df.apply(extract_color, axis=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

print("Loading patrickjohncyh/fashion-clip from Hugging Face...")
model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip').to(device)
processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')
model.eval()

# 2. Text embeddings
print("Extracting text embeddings...")
text_inputs = []
for idx, row in df.iterrows():
    text_input = f"{row['gender']} {row['category']} {row['color']} {row['occasion']} {row['name']} {row['description']}"
    text_inputs.append(text_input)

text_features_list = []
batch_size = 16
for i in range(0, len(text_inputs), batch_size):
    batch_texts = text_inputs[i:i+batch_size]
    inputs = processor(text=batch_texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    text_features_list.append(features.cpu().numpy())

text_embeddings = np.vstack(text_features_list)
np.save('text_embeddings.npy', text_embeddings)
print("Saved text_embeddings.npy. Shape:", text_embeddings.shape)

# 3. Visual embeddings
print("Extracting visual embeddings...")
visual_features_list = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Visual Extraction"):
    img_path = row['image']
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        img = Image.new('RGB', (224, 224), (255, 255, 255))
        print(f"Error loading image {img_path}: {e}")
        
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    visual_features_list.append(features.cpu().numpy())

visual_embeddings = np.vstack(visual_features_list)
np.save('visual_embeddings.npy', visual_embeddings)
print("Saved visual_embeddings.npy. Shape:", visual_embeddings.shape)

# 4. L2-normalize, average, L2-normalize again, compute similarity matrix
def l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

text_norm = l2_normalize(text_embeddings)
visual_norm = l2_normalize(visual_embeddings)

clip_combined = (text_norm + visual_norm) / 2.0
clip_combined_norm = l2_normalize(clip_combined)

sim_matrix = np.dot(clip_combined_norm, clip_combined_norm.T)
np.save('sim_matrix.npy', sim_matrix)
print("Saved sim_matrix.npy. Shape:", sim_matrix.shape)

print("FashionCLIP pre-extraction complete!")
