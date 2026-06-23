import pandas as pd
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pillow_avif
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

# 2. Text embeddings
print("Extracting text embeddings...")
model_text = SentenceTransformer('all-MiniLM-L6-v2')
text_inputs = []
for idx, row in df.iterrows():
    text_input = f"{row['gender']} {row['category']} {row['color']} {row['occasion']} {row['name']} {row['description']}"
    text_inputs.append(text_input)

text_embeddings = model_text.encode(text_inputs, show_progress_bar=True)
np.save('text_embeddings.npy', text_embeddings)
print("Saved text_embeddings.npy with shape:", text_embeddings.shape)

# 3. Visual embeddings
print("Extracting visual embeddings...")
# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load ResNet50
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # Strip final pooling & FC layer
resnet = resnet.to(device)
resnet.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FashionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image']
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to a blank image if load fails
            img = Image.new('RGB', (224, 224), (255, 255, 255))
            print(f"Error loading image {img_path}: {e}")
        if self.transform:
            img = self.transform(img)
        return img

dataset = FashionDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

visual_embs = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Visual extraction"):
        batch = batch.to(device)
        features = resnet(batch) # shape: [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1) # shape: [B, 2048]
        visual_embs.append(features.cpu().numpy())

visual_embeddings = np.vstack(visual_embs)
np.save('visual_embeddings.npy', visual_embeddings)
print("Saved visual_embeddings.npy with shape:", visual_embeddings.shape)

print("Pre-extraction complete!")
