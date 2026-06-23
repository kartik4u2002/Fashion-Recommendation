import nbformat as nbf
import os

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Define cells list
cells = []

# Section 0: Setup & Install
cells.append(nbf.v4.new_markdown_cell("""## SECTION 0: Setup & Install
Ensure you have all the required libraries installed. In Google Colab, we recommend using a T4 GPU runtime for faster model inference.
To enable GPU: **Runtime > Change runtime type > T4 GPU**."""))

cells.append(nbf.v4.new_code_cell("""# Setup dependencies
!pip install transformers sentence-transformers faiss-cpu torch torchvision matplotlib seaborn pandas numpy pillow-heif pillow-avif-plugin scikit-learn ipywidgets tqdm

import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU. In Google Colab, enable T4 GPU: Runtime > Change runtime type > T4 GPU")

# Explicitly import pillow_avif to register AVIF format support for PIL.Image
import pillow_avif
print("pillow-avif-plugin registered successfully for AVIF image support!")"""))


# Section 1: Dataset Analysis & EDA
cells.append(nbf.v4.new_markdown_cell("""## SECTION 1: Dataset Analysis & EDA
In this section, we clone the dataset repository (if running in Colab), load the metadata, handle color extraction from unstructured text descriptions, and visualize distributions and sample images."""))

cells.append(nbf.v4.new_code_cell("""import os
# Clone the repo if we are in Colab (not already cloned)
if not os.path.exists('products.csv') and not os.path.exists('images'):
    print("Cloning repository in Colab...")
    !git clone https://github.com/DarexAI-AI-Startup/ML-TASK.git
    import os
    if os.path.exists('ML-TASK'):
        os.chdir('ML-TASK')

# Inspect folder contents
print("Current Working Directory:", os.getcwd())
for root, dirs, files in os.walk('.'):
    # Exclude git directories for clean output
    if '.git' in root:
        continue
    for f in files:
        print(os.path.join(root, f))"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load data files
df = pd.read_csv('products.csv')
outfits_df = pd.read_csv('outfits.csv')

# Print core metadata characteristics
print("=== products.csv ===")
print("Shape:", df.shape)
print("\\nColumns & Dtypes:")
print(df.dtypes)
print("\\nNull Count per Column:")
print(df.isnull().sum())
print("\\nUnique Values count:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")"""))

cells.append(nbf.v4.new_code_cell("""# Extract colors from name and description using a target color vocabulary
def extract_color(row):
    p_id = row['id']
    # Specific hardcoded rules for items with no color specified in text
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

df['color'] = df.apply(extract_color, axis=1)
print("Color extraction counts:")
print(df['color'].value_counts())"""))

cells.append(nbf.v4.new_code_cell("""# Plotting distributions
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(15, 18))

# 1. Category distribution
sns.countplot(data=df, x='category', ax=axes[0], order=df['category'].value_counts().index, palette='viridis')
axes[0].set_title('Product Category Distribution', fontsize=14, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# 2. Color distribution
sns.countplot(data=df, x='color', ax=axes[1], order=df['color'].value_counts().index, palette='rocket')
axes[1].set_title('Product Color Distribution', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# 3. Occasion / Gender distribution
sns.countplot(data=df, x='occasion', hue='gender', ax=axes[2], palette='mako')
axes[2].set_title('Product Distribution by Occasion & Gender', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 5x5 grid of sample product images with names overlaid
# Using a robust try/except per cell to prevent any image loading error from crashing the EDA plot
fig, axes = plt.subplots(5, 5, figsize=(15, 15))
sample_items = df.sample(25, random_state=42).reset_index(drop=True)

for i in range(25):
    row_idx = i // 5
    col_idx = i % 5
    ax = axes[row_idx, col_idx]
    
    img_path = sample_items.iloc[i]['image']
    name = sample_items.iloc[i]['name']
    if len(name) > 20:
        name = name[:17] + "..."
    category = sample_items.iloc[i]['category']
    
    try:
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error Loading:\\n{os.path.basename(img_path)}", 
                ha='center', va='center', color='red', fontsize=8)
        ax.set_facecolor('#ffe6e6')
        
    ax.set_title(f"{name}\\n({category})", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Image dimensions variance
widths, heights = [], []
for idx, row in df.iterrows():
    try:
        with Image.open(row['image']) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        pass

print("=== Image Dimensions Summary ===")
print(f"Widths: min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.1f}")
print(f"Heights: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.1f}")"""))

cells.append(nbf.v4.new_markdown_cell("""### SECTION 1: EDA Findings & Dataset Characteristics

- **Total Item Count**: The dataset contains 68 unique product items and 25 curated outfits.
- **Category Breakdown**: High diversity with 47 unique categories (e.g. formal-shirts, tshirts, running-shoes, wedding-sarees).
- **Color Palette Distribution**: Top colors are Red (9), White (8), Black (8), Navy Blue (8), Brown (8), and Grey (4).
- **Dataset Challenges**:
  * **Class Imbalance**: With 68 items and 47 categories, many categories contain only a single product (e.g., blazer, cap). This requires broad category groupings and rule-based compatibility rather than deep multi-class classification.
  * **Missing Metadata**: No explicit "color" column was present in `products.csv`. We successfully extracted colors by string-matching product names and descriptions.
  * **AVIF Image Formats**: 28 of the 68 product images are encoded as AVIF format but saved with `.jpg` extensions. Standard Pillow (`PIL.Image`) fails to identify them, requiring `pillow-avif-plugin` and explicit imports to prevent crashes.
  * **Size Reality Check**: With N=68, vector indexes like FAISS are technically overkill (brute-force cosine similarity works instantaneously), but FAISS is built as a demonstration of production-scale architectures.
- **Image Resolution Variance**:
  * Width ranges from 256px to 420px (Mean: 370.8px).
  * Height ranges from 341px to 560px (Mean: 494.3px)."""))


# Section 2: Feature Extraction Pipeline
cells.append(nbf.v4.new_markdown_cell("""## SECTION 2: Feature Extraction Pipeline
Here, we extract text features and visual features using the pretrained **FashionCLIP** model (`patrickjohncyh/fashion-clip`). We combine both into a hybrid multimodal embedding space and build a FAISS index.
⚠️ **Colab Note**: Loading the model and processing the images/metadata takes approximately 1-2 minutes on CPU or 20-30 seconds on GPU."""))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import os
import pillow_avif

# Check for pre-saved embeddings to speed up notebook execution during local runs
USE_PRECOMPUTED = os.path.exists('text_embeddings.npy') and os.path.exists('visual_embeddings.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

print("Loading patrickjohncyh/fashion-clip from Hugging Face...")
model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip').to(device)
processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')
model.eval()

# 2a. Text Feature Extraction
if USE_PRECOMPUTED:
    print("Loading precomputed text embeddings from cache...")
    text_embeddings = np.load('text_embeddings.npy')
else:
    print("Extracting text features using FashionCLIP text encoder...")
    text_inputs = [
        f"{row['gender']} {row['category']} {row['color']} {row['occasion']} {row['name']} {row['description']}"
        for _, row in df.iterrows()
    ]
    
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

print("Text Embeddings shape:", text_embeddings.shape)"""))

cells.append(nbf.v4.new_code_cell("""# 2b. Visual Feature Extraction
if USE_PRECOMPUTED:
    print("Loading precomputed visual embeddings from cache...")
    visual_embeddings = np.load('visual_embeddings.npy')
else:
    print("Extracting visual features using FashionCLIP vision encoder...")
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

print("Visual Embeddings shape:", visual_embeddings.shape)"""))

cells.append(nbf.v4.new_code_cell("""# 2c. Hybrid Embedding and FAISS indexing
import faiss
import os

# Check for precomputed similarity matrix cache to avoid re-extracting
USE_SIM_CACHE = os.path.exists('sim_matrix.npy')

# L2 normalize helper
def l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

# L2-normalize individual embedding modalities
text_norm = l2_normalize(text_embeddings)
visual_norm = l2_normalize(visual_embeddings)

# Compute multimodal hybrid embedding: average of L2-normalized text and visual features
clip_combined = (text_norm + visual_norm) / 2.0
combined_norm = l2_normalize(clip_combined)

if USE_SIM_CACHE:
    print("Loading precomputed similarity matrix from cache...")
    sim_matrix = np.load('sim_matrix.npy')
else:
    print("Computing similarity matrix...")
    sim_matrix = np.dot(combined_norm, combined_norm.T)
    np.save('sim_matrix.npy', sim_matrix)

# Build FAISS IndexFlatIP
dim = combined_norm.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(combined_norm.astype('float32'))

# Save index
faiss.write_index(index, 'fashion.index')
print("Built and saved FAISS IndexFlatIP index with dimension:", dim)
print("Similarity matrix shape:", sim_matrix.shape)"""))


# Section 3: Outfit Compatibility Engine
cells.append(nbf.v4.new_markdown_cell("""## SECTION 3: Outfit Compatibility Engine
In this section, we define the styling compatibility matrix, the color harmony rules, and build a composite pairwise compatibility scorer and outfit generator."""))

cells.append(nbf.v4.new_code_cell("""# 3a. Category Compatibility Matrix
# Mapping each category to list of compatible categories.
COMPATIBLE_CATEGORIES = {
    'formal-shirts': ['trousers', 'chinos', 'blazers', 'suits', 'formal-shoes', 'loafers', 'watches', 'sunglasses'],
    'linen-shirts': ['chinos', 'shorts', 'jeans', 'sandals', 'loafers', 'sunglasses', 'watches'],
    'casual-shirts': ['jeans', 'chinos', 'shorts', 'sneakers', 'boots', 'sandals', 'watches', 'sunglasses'],
    'party-shirts': ['trousers', 'chinos', 'blazers', 'formal-shoes', 'loafers', 'watches'],
    'tshirts': ['jeans', 'chinos', 'shorts', 'track-pants', 'denim-jackets', 'sneakers', 'sandals', 'caps'],
    'polo-tshirts': ['chinos', 'shorts', 'jeans', 'sneakers', 'loafers', 'watches', 'sunglasses'],
    'sweatshirts': ['track-pants', 'jeans', 'shorts', 'sneakers', 'caps'],
    'sweaters': ['jeans', 'chinos', 'trousers', 'boots', 'sneakers', 'watches'],
    'suits': ['formal-shirts', 'formal-shoes', 'loafers', 'watches'],
    'blazers': ['formal-shirts', 'trousers', 'chinos', 'formal-shoes', 'loafers', 'watches'],
    'nehru-jackets': ['kurta-sets', 'sherwanis', 'ethnic-footwear'],
    'denim-jackets': ['tshirts', 'jeans', 'sneakers', 'caps'],
    'long-coats': ['trousers', 'formal-shirts', 'formal-shoes', 'boots'],
    'sherwanis': ['ethnic-footwear', 'nehru-jackets'],
    'kurta-sets': ['ethnic-footwear', 'nehru-jackets'],
    'sharara-sets': ['ethnic-footwear', 'earrings', 'necklaces'],
    'salwar-suits': ['flats', 'sandals', 'ethnic-footwear', 'earrings', 'necklaces'],
    'wedding-sarees': ['flats', 'sandals', 'earrings', 'necklaces', 'clutches'],
    'party-dresses': ['heels', 'sandals', 'clutches', 'earrings', 'necklaces'],
    'casual-dresses': ['flats', 'sandals', 'sneakers', 'handbags', 'earrings', 'sunglasses'],
    'maxi-dresses': ['flats', 'sandals', 'handbags', 'earrings', 'sunglasses'],
    'co-ord-sets': ['flats', 'sandals', 'sneakers', 'handbags', 'earrings'],
    'tops': ['jeans', 'skirts', 'shorts', 'trousers', 'heels', 'sandals', 'flats', 'handbags'],
    'skirts': ['tops', 'tshirts', 'heels', 'sandals', 'flats', 'handbags'],
    'trousers': ['formal-shirts', 'party-shirts', 'blazers', 'formal-shoes', 'loafers', 'watches'],
    'jeans': ['tshirts', 'casual-shirts', 'polo-tshirts', 'sweaters', 'denim-jackets', 'sneakers', 'boots', 'sandals'],
    'chinos': ['formal-shirts', 'casual-shirts', 'polo-tshirts', 'blazers', 'sneakers', 'loafers', 'watches'],
    'shorts': ['tshirts', 'casual-shirts', 'polo-tshirts', 'sneakers', 'sandals'],
    'track-pants': ['tshirts', 'sweatshirts', 'activewear', 'running-shoes'],
    'activewear': ['track-pants', 'running-shoes', 'caps'],
    'running-shoes': ['track-pants', 'activewear', 'tshirts'],
    'sneakers': ['jeans', 'chinos', 'shorts', 'tshirts', 'casual-shirts', 'polo-tshirts', 'sweatshirts', 'denim-jackets'],
    'formal-shoes': ['trousers', 'formal-shirts', 'blazers', 'suits', 'watches'],
    'loafers': ['chinos', 'jeans', 'formal-shirts', 'casual-shirts', 'polo-tshirts', 'blazers', 'watches'],
    'boots': ['jeans', 'casual-shirts', 'sweaters', 'denim-jackets'],
    'sandals': ['jeans', 'shorts', 'casual-shirts', 'linen-shirts', 'casual-dresses', 'maxi-dresses', 'skirts'],
    'flats': ['salwar-suits', 'wedding-sarees', 'casual-dresses', 'maxi-dresses', 'co-ord-sets', 'skirts'],
    'heels': ['party-dresses', 'casual-dresses', 'skirts', 'tops'],
    'ethnic-footwear': ['kurta-sets', 'sherwanis', 'sharara-sets', 'salwar-suits', 'nehru-jackets'],
    'necklaces': ['party-dresses', 'wedding-sarees', 'sharara-sets', 'salwar-suits'],
    'earrings': ['party-dresses', 'casual-dresses', 'maxi-dresses', 'wedding-sarees', 'sharara-sets', 'salwar-suits', 'co-ord-sets'],
    'clutches': ['party-dresses', 'wedding-sarees'],
    'handbags': ['casual-dresses', 'maxi-dresses', 'co-ord-sets', 'tops', 'skirts'],
    'sunglasses': ['casual-shirts', 'linen-shirts', 'tshirts', 'polo-tshirts', 'casual-dresses', 'maxi-dresses'],
    'watches': ['formal-shirts', 'casual-shirts', 'party-shirts', 'polo-tshirts', 'blazers', 'suits', 'trousers', 'chinos'],
    'caps': ['tshirts', 'sweatshirts', 'activewear', 'track-pants', 'jeans', 'shorts']
}

# Fill missing rules to avoid KeyErrors
for cat in df['category'].unique():
    if cat not in COMPATIBLE_CATEGORIES:
        COMPATIBLE_CATEGORIES[cat] = []"""))

cells.append(nbf.v4.new_code_cell("""# 3b. Color Harmony Rules
NEUTRAL_COLORS = ["White", "Black", "Grey", "Navy Blue", "Beige", "Off White", "Cream", "Dark Grey"]

COLOR_HARMONY = {
    'neutrals': NEUTRAL_COLORS,
    'pairs': {
        frozenset(["Red", "Black"]),
        frozenset(["Red", "White"]),
        frozenset(["Red", "Grey"]),
        frozenset(["Blue", "Beige"]),
        frozenset(["Blue", "White"]),
        frozenset(["Green", "Beige"]),
        frozenset(["Green", "White"]),
        frozenset(["Pink", "Grey"]),
        frozenset(["Pink", "White"]),
        frozenset(["Yellow", "Navy Blue"]),
        frozenset(["Yellow", "White"]),
        frozenset(["Gold", "Black"]),
        frozenset(["Gold", "Red"]),
        frozenset(["Lavender", "Blue"]),
        frozenset(["Lavender", "White"]),
        frozenset(["Lavender", "Grey"]),
        frozenset(["Olive", "Black"]),
        frozenset(["Olive", "Beige"]),
        frozenset(["Maroon", "Black"]),
        frozenset(["Maroon", "White"]),
        frozenset(["Cream", "Gold"]),
        frozenset(["Tan", "Navy Blue"]),
        frozenset(["Tan", "White"])
    }
}

def color_compatible(c1, c2) -> bool:
    \"\"\"
    Checks if two colors are compatible.
    Rules: same color = True, neutral + anything = True, specified harmonious pairs = True
    \"\"\"
    if c1 == 'Unknown' or c2 == 'Unknown':
        return True
    if c1 == c2:
        return True
    if c1 in COLOR_HARMONY['neutrals'] or c2 in COLOR_HARMONY['neutrals']:
        return True
    if frozenset([c1, c2]) in COLOR_HARMONY['pairs']:
        return True
    return False"""))

cells.append(nbf.v4.new_code_cell("""# 3c. Pairwise Compatibility Scorer
def compatibility_score(id1_or_idx, id2_or_idx, df, sim_matrix) -> tuple:
    \"\"\"
    Computes compatibility score in [0.0, 1.0] and returns an explanation.
    Supports both string product IDs and integer indices.
    Formula: Score = 0.4*category_match + 0.3*color_harmony + 0.3*embedding_similarity
    \"\"\"
    # Robust index lookup with error handling
    if isinstance(id1_or_idx, (int, np.integer)):
        idx1 = int(id1_or_idx)
    else:
        matches = df[df['id'] == id1_or_idx]
        if matches.empty:
            raise ValueError(f"Item ID '{id1_or_idx}' not found in dataframe")
        idx1 = matches.index[0]
        
    if isinstance(id2_or_idx, (int, np.integer)):
        idx2 = int(id2_or_idx)
    else:
        matches = df[df['id'] == id2_or_idx]
        if matches.empty:
            raise ValueError(f"Item ID '{id2_or_idx}' not found in dataframe")
        idx2 = matches.index[0]
    
    row1 = df.iloc[idx1]
    row2 = df.iloc[idx2]
    
    # 1. Category matching
    cat1, cat2 = row1['category'], row2['category']
    cat_match = 0.0
    if cat2 in COMPATIBLE_CATEGORIES.get(cat1, []) or cat1 in COMPATIBLE_CATEGORIES.get(cat2, []):
        cat_match = 1.0
        
    # 2. Color harmony matching
    col1, col2 = row1['color'], row2['color']
    col_match = 1.0 if color_compatible(col1, col2) else 0.0
    
    # 3. Embedding similarity from precomputed sim_matrix
    sim = sim_matrix[idx1, idx2]
    sim_score = float(np.clip((sim + 1) / 2, 0.0, 1.0)) # Scale to [0, 1]
    
    total_score = 0.4 * cat_match + 0.3 * col_match + 0.3 * sim_score
    
    # Explanation builder
    reasons = []
    if cat_match > 0:
        reasons.append(f"the style profile of {cat1} pairs well with {cat2}")
    else:
        reasons.append(f"{cat1} and {cat2} are an unconventional clothing combination")
        
    if col_match > 0:
        if col1 == col2:
            reasons.append(f"the monochromatic look in {col1} is clean and structured")
        elif col1 in COLOR_HARMONY['neutrals'] or col2 in COLOR_HARMONY['neutrals']:
            reasons.append(f"the classic contrast of {col1} and {col2} is versatile")
        else:
            reasons.append(f"the color pairing of {col1} and {col2} looks naturally balanced")
    else:
        reasons.append(f"the color clash between {col1} and {col2} creates a highly bold aesthetic")
        
    reasons.append(f"the visual-textual similarity match is {sim_score*100:.0f}%")
    explanation = f"Recommended because " + ", and ".join(reasons) + "."
    
    return total_score, explanation"""))

cells.append(nbf.v4.new_code_cell("""# 3d. Outfit Generator
# Define OUTFIT_SLOTS explicitly first
OUTFIT_SLOTS = {
    'top': ['formal-shirts', 'casual-shirts', 'linen-shirts', 'party-shirts', 'tshirts', 'polo-tshirts', 'sweatshirts', 'sweaters', 'tops'],
    'bottom': ['trousers', 'jeans', 'chinos', 'shorts', 'track-pants', 'skirts', 'leggings'],
    'footwear': ['running-shoes', 'sneakers', 'ethnic-footwear', 'heels', 'boots', 'flats', 'formal-shoes', 'loafers', 'sandals'],
    'layer': ['blazers', 'suits', 'nehru-jackets', 'denim-jackets', 'long-coats'],
    'dress': ['party-dresses', 'casual-dresses', 'maxi-dresses', 'co-ord-sets', 'kurta-sets', 'sherwanis', 'sharara-sets', 'salwar-suits', 'wedding-sarees'],
    'accessory': ['necklaces', 'earrings', 'clutches', 'handbags', 'sunglasses', 'watches', 'caps']
}

# Dynamically invert mapping: {article_type: slot_name}
SLOT_MAPPING = {article_type: slot_name for slot_name, types in OUTFIT_SLOTS.items() for article_type in types}

def generate_complete_outfit(seed_id, df, sim_matrix, top_k=10):
    \"\"\"
    Generates a complete outfit around a seed product ID.
    Recommends items for required slots: bottom, footwear, accessory.
    \"\"\"
    # Robust index lookup with error handling
    matches = df[df['id'] == seed_id]
    if matches.empty:
        raise ValueError(f"Seed ID '{seed_id}' not found in dataframe")
    seed_idx = matches.index[0]
    seed_item = df.iloc[seed_idx]
    seed_slot = SLOT_MAPPING.get(seed_item['category'], 'accessory')
    seed_gender = seed_item['gender']
    seed_occasion = seed_item['occasion']
    
    if seed_slot == 'dress':
        target_slots = ['footwear', 'accessory']
    elif seed_slot == 'top':
        target_slots = ['bottom', 'footwear', 'accessory']
    elif seed_slot == 'bottom':
        target_slots = ['top', 'footwear', 'accessory']
    elif seed_slot == 'footwear':
        target_slots = ['top', 'bottom', 'accessory']
    else:
        target_slots = ['top', 'bottom', 'footwear']
        
    complementary_items = []
    
    for slot in target_slots:
        best_cand = None
        best_score = -1
        best_explanation = ""
        
        # Gather matching items from the dataset
        for idx, row in df.iterrows():
            # Gap 2: Exclude seed item explicitly
            if idx == seed_idx:
                continue
            item_slot = SLOT_MAPPING.get(row['category'], 'accessory')
            
            # Special category override: dress can act as a top for outfit slot matching
            if slot == 'top' and item_slot == 'dress':
                pass
            elif item_slot != slot:
                continue
                
            # Filter gender
            if seed_gender != 'unisex' and row['gender'] != 'unisex' and row['gender'] != seed_gender:
                continue
                
            score, explanation = compatibility_score(seed_id, row['id'], df, sim_matrix)
            
            # Boost score slightly if occasion matches
            if row['occasion'] == seed_occasion:
                score += 0.05
                explanation += f" This item matches the '{seed_occasion}' occasion requirements."
            score = float(np.clip(score, 0.0, 1.0))
            
            if score > best_score:
                best_score = score
                best_cand = row
                best_explanation = explanation
                
        if best_cand is not None:
            complementary_items.append({
                'slot': slot,
                'item': best_cand.to_dict(),
                'score': round(best_score, 2),
                'reason': best_explanation
            })
            
    return {
        'seed': seed_item.to_dict(),
        'complementary_items': complementary_items
    }"""))


# Section 4: Chat-based Fashion Assistant
cells.append(nbf.v4.new_markdown_cell("""## SECTION 4: Chat-based Fashion Assistant
A conversational assistant interface using a keyword parser, style response builder, and ipywidgets with a robust print fallback."""))

cells.append(nbf.v4.new_code_cell("""# 4a. Natural Language Parser
def parse_intent(user_text: str, df) -> dict:
    \"\"\"
    Extracts intent and filters (category, color, gender, occasion) from raw text.
    \"\"\"
    text = user_text.lower()
    
    # Intent mapping
    intent = 'find_item'
    if any(kw in text for kw in ["outfit", "wear", "pair with", "match", "goes with", "style", "coordinate", "combine"]):
        intent = 'complete_outfit'
    elif any(kw in text for kw in ["does", "compatible", "go well with"]):
        intent = 'compatibility_check'
    elif any(kw in text for kw in ["advice", "tip", "should i wear", "trend"]):
        intent = 'style_advice'
        
    filters = {'category': None, 'color': None, 'gender': None, 'occasion': None}
    
    # Gender extraction
    if any(kw in text for kw in ['women', 'female', 'girl', 'her']):
        filters['gender'] = 'women'
    elif any(kw in text for kw in ['men', 'male', 'guy', 'him']):
        filters['gender'] = 'men'
        
    # Occasion extraction
    for occ in df['occasion'].unique():
        if occ.lower() in text:
            filters['occasion'] = occ
            break
            
    # Color extraction
    colors = ["white", "black", "grey", "gray", "navy blue", "navy", "beige", "off white", "blue", "red", "green", "yellow", "pink", "purple", "orange", "brown", "gold", "silver", "olive", "maroon", "peach", "cream", "teal", "khaki", "rust", "mustard", "tan", "wine", "turquoise", "lilac", "lavender", "rose"]
    for col in colors:
        if col in text:
            val = col.title()
            if val == 'Gray':
                val = 'Grey'
            if val == 'Navy':
                val = 'Navy Blue'
            filters['color'] = val
            break
            
    # Category synonyms and extraction
    synonyms = {
        'shirt': 'formal-shirts', 'jean': 'jeans', 'pant': 'trousers', 'trouser': 'trousers',
        'chino': 'chinos', 'dress': 'party-dresses', 'skirt': 'skirts', 'suit': 'suits',
        'blazer': 'blazers', 'sneaker': 'sneakers', 'shoe': 'sneakers', 'heel': 'heels',
        'saree': 'wedding-sarees', 'kurta': 'kurta-sets', 'jacket': 'denim-jackets'
    }
    
    found_cat = False
    for cat in df['category'].unique():
        if cat.replace('-', ' ') in text or cat in text:
            filters['category'] = cat
            found_cat = True
            break
    if not found_cat:
        for kw, cat in synonyms.items():
            if kw in text:
                filters['category'] = cat
                break
                
    # Direct ID extraction
    item_id = None
    for idx, row in df.iterrows():
        short_id = str(row['id']).split('_')[-1]
        if short_id in text or str(row['id']) in text:
            item_id = row['id']
            break
            
    return {
        'intent': intent,
        'filters': filters,
        'item_id': item_id,
        'query_text': user_text
    }"""))

cells.append(nbf.v4.new_code_cell("""# 4b. Response Builder
def build_response(intent, parsed, df, sim_matrix) -> tuple:
    \"\"\"
    Builds emoji-rich conversational styling responses.
    \"\"\"
    filters = parsed['filters']
    item_id = parsed['item_id']
    outfit_items = []
    
    if intent == 'complete_outfit':
        # Select best seed item matching filters
        candidates = df.copy()
        for key, val in filters.items():
            if val:
                candidates = candidates[candidates[key] == val]
        
        if candidates.empty:
            if item_id:
                seed = df[df['id'] == item_id].iloc[0]
            else:
                seed = df.sample(1).iloc[0]
        else:
            seed = candidates.iloc[0]
            
        outfit = generate_complete_outfit(seed['id'], df, sim_matrix)
        seed_dict = outfit['seed']
        comps = outfit['complementary_items']
        
        outfit_items.append(seed_dict)
        response = f"✨ **Stylist Assistant**: I've designed a gorgeous look for you around the **{seed_dict['brand']} {seed_dict['name']}** in {seed_dict['color']}!\\n\\n"
        response += f"👚 **Hero Item**: {seed_dict['name']} (₹{seed_dict['price_inr']})\\n"
        
        for comp in comps:
            item = comp['item']
            response += f"\\n👉 **Slot [{comp['slot'].upper()}]**: {item['brand']} {item['name']} (Compatibility: {comp['score']*100:.0f}%)\\n"
            response += f"   💬 *Rationale*: {comp['reason']}\\n"
            outfit_items.append(item)
            
        response += f"\\n🎨 **Stylist Tip**: This curated set captures the perfect color harmony for your {seed_dict['occasion']} occasion."
        return response, outfit_items
        
    elif intent == 'find_item':
        query_text = parsed.get('query_text', '')
        candidates = df.copy()
        for key, val in filters.items():
            if val:
                candidates = candidates[candidates[key] == val]
                
        use_clip_search = False
        if query_text:
            try:
                inputs = processor(text=[query_text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
                with torch.no_grad():
                    query_emb = model.get_text_features(**inputs).cpu().numpy()
                query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
                similarities = np.dot(visual_norm, query_norm.T).squeeze()
                
                df_temp = df.copy()
                df_temp['similarity'] = similarities
                
                if not candidates.empty:
                    candidates = df_temp.loc[candidates.index].sort_values(by='similarity', ascending=False)
                else:
                    candidates = df_temp.sort_values(by='similarity', ascending=False)
                use_clip_search = True
            except Exception as e:
                # If CLIP variables aren't globally available or fail, we print error and use fallback
                pass
                
        if use_clip_search:
            response = "🔍 **Stylist Assistant**: Here are the matching products I found for you using zero-shot semantic search:\\n"
            for idx, row in candidates.head(3).iterrows():
                sim_score = row['similarity']
                response += f"\\n- **{row['brand']} {row['name']}** in {row['color']} (₹{row['price_inr']}) [Match: {sim_score*100:.1f}%]"
                outfit_items.append(df.loc[idx].to_dict())
            return response, outfit_items
        else:
            if candidates.empty:
                response = "🔍 **Stylist Assistant**: I couldn't find products matching that exact search. Check out these highly popular alternatives:\\n"
                samples = df.sample(min(3, len(df)))
                for row in samples.itertuples():
                    response += f"\\n- **{row.brand} {row.name}** in {row.color} (₹{row.price_inr})"
                    outfit_items.append(df.loc[row.Index].to_dict())
                return response, outfit_items
                
            response = f"🔍 **Stylist Assistant**: Here are the matching products I found for you:\\n"
            for row in candidates.head(3).itertuples():
                response += f"\\n- **{row.brand} {row.name}** in {row.color} (₹{row.price_inr}) - perfect choice for a {row.occasion}!"
                outfit_items.append(df.loc[row.Index].to_dict())
            return response, outfit_items
        
    elif intent == 'compatibility_check':
        if item_id:
            seed = df[df['id'] == item_id].iloc[0]
            outfit = generate_complete_outfit(seed['id'], df, sim_matrix)
            if outfit['complementary_items']:
                comp = outfit['complementary_items'][0]
                item2 = comp['item']
                score = comp['score']
                reason = comp['reason']
            else:
                item2 = df.sample(1).iloc[0].to_dict()
                score, reason = compatibility_score(seed['id'], item2['id'], df, sim_matrix)
                
            response = f"🤔 **Stylist Assistant**: Checking styling compatibility between {seed['name']} and {item2['name']}...\\n\\n"
            response += f"📊 **Compatibility Rating**: {score*100:.0f}%\\n"
            response += f"📝 **Stylist Verdict**: {reason}"
            outfit_items.extend([seed.to_dict(), item2])
            return response, outfit_items
        else:
            p1 = df.sample(1).iloc[0]
            outfit = generate_complete_outfit(p1['id'], df, sim_matrix)
            if outfit['complementary_items']:
                comp = outfit['complementary_items'][0]
                p2 = comp['item']
                score = comp['score']
                reason = comp['reason']
            else:
                p2 = df.sample(1).iloc[0].to_dict()
                score, reason = compatibility_score(p1['id'], p2['id'], df, sim_matrix)
                
            response = f"🤔 **Stylist Assistant**: Checking if the **{p1['brand']} {p1['name']}** pairs well with the **{p2['brand']} {p2['name']}**...\\n\\n"
            response += f"📊 **Compatibility Rating**: {score*100:.0f}%\\n"
            response += f"📝 **Stylist Verdict**: {reason}"
            outfit_items.extend([p1.to_dict(), p2])
            return response, outfit_items
            
    else: # style_advice
        response = "💡 **Stylist Assistant**: Here are 3 essential styling rules:\\n\\n" \\
                   "1. **Rule of Neutrals**: Ground your outfit with a neutral base (Black, White, Beige, Grey, Navy).\\n" \
                   "2. **Balancing Fits**: Combine form-fitting silhouettes with relaxed fits for optimal visual weight.\\n" \
                   "3. **Footwear Anchoring**: Footwear should match the occasion context (e.g. sneakers for sports, heels for parties)."
        return response, []"""))

cells.append(nbf.v4.new_code_cell("""# 4c. Interactive Chat Widget (ipywidgets)
import ipywidgets as widgets
from IPython.display import display, HTML

chat_history = []

def format_bubble(sender, text):
    if sender == 'user':
        return f"<div style='text-align: right; margin: 5px;'><span style='background-color: #d1e7dd; color: #0f5132; padding: 8px 12px; border-radius: 15px; display: inline-block; max-width: 70%; font-family: sans-serif;'>{text}</span></div>"
    else:
        text_html = text.replace('\\n', '<br>').replace('**', '<b>').replace('✨', '✨').replace('👉', '👉').replace('💬', '💬')
        return f"<div style='text-align: left; margin: 5px;'><span style='background-color: #f1f3f5; color: #212529; padding: 8px 12px; border-radius: 15px; display: inline-block; max-width: 70%; font-family: sans-serif;'>{text_html}</span></div>"

output_area = widgets.Output()
text_input = widgets.Text(placeholder="Message the stylist assistant...", layout=widgets.Layout(width='80%'))
send_btn = widgets.Button(description="Send", button_style='primary')

def display_images_horizontal(items):
    if not items:
        return
    html = "<div style='display: flex; flex-direction: row; flex-wrap: wrap; gap: 10px; margin-top: 10px; justify-content: flex-start;'>"
    for item in items[:3]:
        img_path = item['image']
        html += f"<div style='border: 1px solid #ddd; padding: 5px; border-radius: 8px; width: 120px; text-align: center; background-color: white;'>"
        html += f"<img src='{img_path}' style='width: 100px; height: 130px; object-fit: cover; border-radius: 4px;' onerror=\\\"this.style.display='none'; this.nextSibling.style.display='block';\\\"><div style='display:none; height:130px; line-height:130px; color:red;'>No Image</div>"
        html += f"<div style='font-size: 9px; font-weight: bold; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{item['brand']}</div>"
        html += f"<div style='font-size: 8px; color: #666; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{item['name']}</div>"
        html += "</div>"
    html += "</div>"
    with output_area:
        display(HTML(html))

def on_send(btn):
    user_text = text_input.value.strip()
    if not user_text:
        return
    text_input.value = ""
    
    with output_area:
        display(HTML(format_bubble('user', user_text)))
        
    chat_history.append(('user', user_text))
    if len(chat_history) > 5:
        chat_history.pop(0)
        
    parsed = parse_intent(user_text, df)
    response_text, items = build_response(parsed['intent'], parsed, df, sim_matrix)
    
    chat_history.append(('assistant', response_text))
    if len(chat_history) > 5:
        chat_history.pop(0)
        
    with output_area:
        display(HTML(format_bubble('assistant', response_text)))
    if items:
        display_images_horizontal(items)

send_btn.on_click(on_send)
text_input.on_submit(on_send)

chat_box = widgets.VBox([output_area, widgets.HBox([text_input, send_btn])])"""))

cells.append(nbf.v4.new_code_cell("""# Uncomment to interact with the widget:
# display(chat_box)

# Fallback: Print-based Simulation (for static rendering on GitHub or initial load)
def chat_simulation():
    print("=== Conversational Fashion Assistant - Static Fallback Demo ===")
    demo_queries = [
        "Show me a complete outfit for a formal meeting",
        "What can I pair with navy blue jeans?",
        "Find me a casual summer dress in white",
        "Does a black shirt go with khaki trousers?"
    ]
    for q in demo_queries:
        print(f"\\n[User]: {q}")
        parsed = parse_intent(q, df)
        response, items = build_response(parsed['intent'], parsed, df, sim_matrix)
        print(response)
        if items:
            print("Recommended Products:", ", ".join([f"{i['brand']} {i['name']}" for i in items]))

chat_simulation()"""))


# Section 5: Evaluation & Visualization
cells.append(nbf.v4.new_markdown_cell("""## SECTION 5: Evaluation & Visualization
Here, we benchmark the outfit compatibility scores against the ground-truth stylist-curated combinations in `outfits.csv`, showcase sample outfit configurations, and plot the 2D PCA representation of our hybrid embeddings."""))

cells.append(nbf.v4.new_code_cell("""# 5a. Quantitative Evaluation using outfits.csv as ground-truth benchmark
# Gap 4 Verification: Call compatibility_score both ways and assert same result
score_by_int = compatibility_score(0, 1, df, sim_matrix)
score_by_str = compatibility_score(df.iloc[0]['id'], df.iloc[1]['id'], df, sim_matrix)
assert score_by_int[0] == score_by_str[0], "Backward compatibility test failed!"
print("✅ Verification Check: compatibility_score backward compatibility confirmed.")

outfit_scores = []
for idx, row in outfits_df.iterrows():
    hero_id = row['hero_id']
    second_id = row['second_id']
    footwear_id = row['footwear_id']
    
    scores = []
    if pd.notnull(second_id) and second_id in df['id'].values and hero_id in df['id'].values:
        s, _ = compatibility_score(hero_id, second_id, df, sim_matrix)
        scores.append(s)
    if pd.notnull(footwear_id) and footwear_id in df['id'].values and hero_id in df['id'].values:
        s, _ = compatibility_score(hero_id, footwear_id, df, sim_matrix)
        scores.append(s)
        
    if scores:
        outfit_scores.append(np.mean(scores))

# Control baseline using random pairings
random_scores = []
np.random.seed(42)
for _ in range(100):
    p1 = df.sample(1).iloc[0]['id']
    p2 = df.sample(1).iloc[0]['id']
    if p1 != p2:
        s, _ = compatibility_score(p1, p2, df, sim_matrix)
        random_scores.append(s)

mean_curated = np.mean(outfit_scores)
std_curated = np.std(outfit_scores)
mean_random = np.mean(random_scores)
pct_curated_high = (np.array(outfit_scores) > 0.70).sum() / len(outfit_scores) * 100

print(f"=== Quantitative Performance Evaluation ===")
print(f"Ground-Truth Curated Outfit Score: {mean_curated:.3f} ± {std_curated:.3f}")
print(f"Random Control Pairs Baseline:   {mean_random:.3f}")
print(f"Percentage of Curated Outfits scoring > 0.7: {pct_curated_high:.1f}%")

plt.figure(figsize=(10, 5))
sns.histplot(outfit_scores, color='green', kde=True, label='Curated Outfits (Stylist)', stat='density', alpha=0.6)
sns.histplot(random_scores, color='red', kde=True, label='Random Pairings (Control)', stat='density', alpha=0.4)
plt.title('Comparison of Compatibility Score Distributions', fontsize=12, fontweight='bold')
plt.xlabel('Compatibility Score')
plt.ylabel('Density')
plt.legend()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 5b. Visual Showcase
diverse_seeds = [
    df[df['category'] == 'formal-shirts'].iloc[0]['id'],
    df[df['category'] == 'party-dresses'].iloc[0]['id'],
    df[df['category'] == 'jeans'].iloc[0]['id'],
    df[df['category'] == 'running-shoes'].iloc[0]['id'],
    df[df['category'] == 'kurta-sets'].iloc[0]['id']
]

for idx, seed_id in enumerate(diverse_seeds):
    outfit = generate_complete_outfit(seed_id, df, sim_matrix)
    seed = outfit['seed']
    comps = outfit['complementary_items']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    # Plot seed
    try:
        img = Image.open(seed['image']).convert('RGB')
        axes[0].imshow(img)
    except Exception:
        axes[0].text(0.5, 0.5, "Image Load\\nFailed", ha='center', va='center')
    axes[0].set_title(f"SEED:\\n{seed['brand']}\\n{seed['name'][:20]}...", fontsize=9, fontweight='bold', color='blue')
    axes[0].axis('off')
    
    # Plot recommendations
    for i in range(3):
        ax = axes[i+1]
        if i < len(comps):
            item = comps[i]['item']
            score = comps[i]['score']
            try:
                img = Image.open(item['image']).convert('RGB')
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image Load\\nFailed", ha='center', va='center')
            ax.set_title(f"Rec {i+1} ({score*100:.0f}%):\\n{item['brand']}\\n{item['name'][:18]}...", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No Item Available", ha='center', va='center')
            ax.set_facecolor('#f0f0f0')
        ax.axis('off')
        
    plt.suptitle(f"Outfit Recommendation Loop {idx+1} for {seed['occasion']} ({seed['gender']})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 5c. Embedding Space Visualization
from sklearn.decomposition import PCA

def get_master_category(cat):
    apparel = ['formal-shirts', 'suits', 'sweatshirts', 'track-pants', 'linen-shirts', 'sherwanis',
               'party-dresses', 'wedding-sarees', 'sharara-sets', 'casual-shirts', 'party-shirts',
               'tshirts', 'polo-tshirts', 'trousers', 'jeans', 'chinos', 'shorts', 'casual-dresses',
               'kurta-sets', 'nehru-jackets', 'skirts', 'activewear', 'denim-jackets', 'sweaters',
               'tops', 'maxi-dresses', 'co-ord-sets', 'leggings', 'salwar-suits', 'long-coats', 'blazers']
    footwear = ['running-shoes', 'sneakers', 'ethnic-footwear', 'heels', 'boots', 'flats', 'formal-shoes',
                'loafers', 'sandals']
    if cat in apparel:
        return 'Apparel'
    elif cat in footwear:
        return 'Footwear'
    else:
        return 'Accessories'

df['masterCategory'] = df['category'].apply(get_master_category)

# Perform PCA to reduce dimensions to 2D
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(combined_norm)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df['masterCategory'], style=df['masterCategory'], s=100, palette='Set1')

# Add category annotations to clusters
for mc in df['masterCategory'].unique():
    indices = df[df['masterCategory'] == mc].index
    mean_x = coords[indices, 0].mean()
    mean_y = coords[indices, 1].mean()
    sample_cat = df.iloc[indices[0]]['category']
    plt.annotate(f"{mc}\\n({sample_cat})", (mean_x, mean_y), textcoords="offset points", 
                 xytext=(0,12), ha='center', fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

plt.title('Fashion Product Embeddings Projection (PCA 2D)', fontsize=14, fontweight='bold')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.tight_layout()
plt.show()"""))


# Section 6: Demo Showcase Cell
cells.append(nbf.v4.new_markdown_cell("""## SECTION 6: Demo Showcase Cell
This section contains three final end-to-end demonstrations run dynamically."""))

cells.append(nbf.v4.new_code_cell("""# 1. Text query -> retrieved items
print("--- DEMO 1: Text query retrieval ---")
demo_query = "casual summer dress in white"
print("Query:", demo_query)
parsed = parse_intent(demo_query, df)
response, items = build_response(parsed['intent'], parsed, df, sim_matrix)
print(response)

# 2. Item ID -> complete outfit grid
print("\\n--- DEMO 2: Complete outfit generation from Seed ID ---")
random_seed_id = df.iloc[10]['id']
print("Seed product ID:", random_seed_id)
outfit = generate_complete_outfit(random_seed_id, df, sim_matrix)
print(f"Hero product: {outfit['seed']['brand']} {outfit['seed']['name']} (Color: {outfit['seed']['color']})")
for idx, comp in enumerate(outfit['complementary_items']):
    print(f"  Rec {idx+1} [{comp['slot'].upper()}]: {comp['item']['brand']} {comp['item']['name']} ({comp['item']['color']}) - Compatibility Score: {comp['score']:.2f}")

# 3. Chat interaction -> 5-turn conversation transcript
print("\\n--- DEMO 3: Simulated Chat Interaction Transcript (5 turns) ---")
conversation_inputs = [
    "I need a complete formal office outfit.",
    "What shoes would match this?",
    "Find a wedding sherwani in beige.",
    "What can I pair with navy blue chinos?",
    "Does a black shirt go with a gold necklace?"
]
for idx, q in enumerate(conversation_inputs):
    print(f"\\nTurn {idx+1} [User]: {q}")
    parsed = parse_intent(q, df)
    response, items = build_response(parsed['intent'], parsed, df, sim_matrix)
    print(response)"""))


# Section 7: User & Context-Aware Recommendations
cells.append(nbf.v4.new_markdown_cell("""## SECTION 7: User & Context-Aware Recommendations
> This section extends the base outfit generator with 4 profile dimensions: Gender, Age Group, Occasion, and Style Preference.
> Recommendations are re-ranked using a 5-signal weighted scorer that combines base compatibility (30%), occasion relevance (25%), color context (20%), style alignment (15%), and age-formality fit (10%)."""))

# 7a: User Profile Schema
cells.append(nbf.v4.new_markdown_cell("""## 7a: User Profile Schema
Define the UserProfile dataclass to represent demographic and situational context."""))

cells.append(nbf.v4.new_code_cell("""from dataclasses import dataclass, field
from typing import Optional

@dataclass
class UserProfile:
    \"\"\"
    Central user profile object for context-aware recommendations.
    All fields are optional — missing fields = no filtering on that dimension.
    \"\"\"
    gender:      Optional[str] = None   # 'Men', 'Women', 'Boys', 'Girls', 'Unisex'
    age_group:   Optional[str] = None   # '20s', '30s', '40s', '50s+'
    occasion:    Optional[str] = None   # 'Office', 'Party', 'Wedding', 'Beach', 
                                        #  'Casual', 'Formal', 'Date Night', 'Sport'
    style_pref:  Optional[str] = None   # 'Formal', 'Smart Casual', 'Casual', 
                                        #  'Streetwear', 'Bohemian', 'Minimalist'
    
    def summary(self) -> str:
        parts = []
        if self.gender:    parts.append(f"Gender: {self.gender}")
        if self.age_group: parts.append(f"Age: {self.age_group}")
        if self.occasion:  parts.append(f"Occasion: {self.occasion}")
        if self.style_pref:parts.append(f"Style: {self.style_pref}")
        return " | ".join(parts) if parts else "No profile set"

# Demo: create test profiles used throughout this section
profile_office_man    = UserProfile(gender='Men',   age_group='30s', 
                                    occasion='Office',  style_pref='Formal')
profile_party_woman   = UserProfile(gender='Women', age_group='20s', 
                                    occasion='Party',   style_pref='Smart Casual')
profile_casual_teen   = UserProfile(gender='Boys',  age_group='20s', 
                                    occasion='Casual',  style_pref='Streetwear')
profile_wedding_woman = UserProfile(gender='Women', age_group='40s', 
                                    occasion='Wedding', style_pref='Formal')

print("Sample profiles created:")
for p in [profile_office_man, profile_party_woman, 
          profile_casual_teen, profile_wedding_woman]:
    print(f"  → {p.summary()}")"""))


# 7b: Context Mapping Tables
cells.append(nbf.v4.new_markdown_cell("""## 7b: Context Mapping Tables
Static lookup tables to match profile dimensions with styling preferences."""))

cells.append(nbf.v4.new_code_cell("""# NEUTRAL_COLORS defined locally to support independent cell execution
NEUTRAL_COLORS = ["White", "Black", "Grey", "Navy Blue", "Beige", "Off White", "Cream", "Dark Grey"]

# OCCASION → article type preferences + color palette
OCCASION_PROFILE = {
    'Office': {
        'preferred_types':  ['formal-shirts', 'linen-shirts', 'trousers', 'chinos', 'blazers', 'suits', 'formal-shoes', 'loafers', 'watches'],
        'preferred_colors': ['White', 'Black', 'Grey', 'Navy Blue', 'Beige', 'Off White'],
        'avoid_types':      ['shorts', 'track-pants', 'activewear', 'caps'],
        'formality_score':  0.9,
    },
    'Party': {
        'preferred_types':  ['party-dresses', 'casual-dresses', 'tops', 'jeans', 'heels', 'clutches', 'earrings', 'necklaces'],
        'preferred_colors': ['Black', 'Red', 'Maroon', 'Gold', 'Silver', 'Navy Blue', 'Pink'],
        'avoid_types':      ['track-pants', 'running-shoes', 'trousers'],
        'formality_score':  0.6,
    },
    'Wedding': {
        'preferred_types':  ['kurta-sets', 'sherwanis', 'sharara-sets', 'salwar-suits', 'wedding-sarees', 'blazers', 'heels', 'flats', 'ethnic-footwear', 'clutches', 'earrings', 'necklaces'],
        'preferred_colors': ['Red', 'Gold', 'Maroon', 'Pink', 'Cream', 'Beige', 'Navy Blue', 'Off White'],
        'avoid_types':      ['jeans', 'tshirts', 'sneakers', 'shorts'],
        'formality_score':  0.95,
    },
    'Beach': {
        'preferred_types':  ['shorts', 'tshirts', 'casual-dresses', 'maxi-dresses', 'sandals', 'flats', 'sunglasses', 'caps'],
        'preferred_colors': ['White', 'Blue', 'Yellow', 'Tan', 'Light Blue', 'Green'],
        'avoid_types':      ['formal-shoes', 'blazers', 'trousers', 'sweaters', 'boots'],
        'formality_score':  0.1,
    },
    'Casual': {
        'preferred_types':  ['tshirts', 'casual-shirts', 'polo-tshirts', 'jeans', 'shorts', 'sneakers', 'loafers', 'caps', 'handbags'],
        'preferred_colors': ['Blue', 'White', 'Grey', 'Black', 'Olive', 'Khaki', 'Red'],
        'avoid_types':      ['formal-shoes', 'blazers', 'trousers'],
        'formality_score':  0.2,
    },
    'Formal': {
        'preferred_types':  ['formal-shirts', 'trousers', 'chinos', 'blazers', 'suits', 'formal-shoes', 'loafers', 'watches'],
        'preferred_colors': ['White', 'Black', 'Navy Blue', 'Dark Grey', 'Grey', 'Maroon'],
        'avoid_types':      ['jeans', 'tshirts', 'sneakers', 'shorts', 'caps'],
        'formality_score':  1.0,
    },
    'Date Night': {
        'preferred_types':  ['casual-shirts', 'party-dresses', 'casual-dresses', 'maxi-dresses', 'jeans', 'heels', 'loafers', 'watches', 'earrings', 'necklaces'],
        'preferred_colors': ['Black', 'Red', 'Maroon', 'Navy Blue', 'White'],
        'avoid_types':      ['track-pants', 'running-shoes', 'caps'],
        'formality_score':  0.65,
    },
    'Sport': {
        'preferred_types':  ['track-pants', 'activewear', 'tshirts', 'shorts', 'running-shoes', 'caps'],
        'preferred_colors': ['Black', 'White', 'Grey', 'Blue', 'Red'],
        'avoid_types':      ['formal-shoes', 'heels', 'formal-shirts', 'party-dresses', 'blazers'],
        'formality_score':  0.05,
    },
}

# AGE GROUP → style modifier weights
AGE_STYLE_WEIGHTS = {
    '20s': {
        'style_boost':  ['Streetwear', 'Casual', 'Smart Casual'],
        'color_boost':  ['Bold', 'Bright'],
        'formality_adj': -0.1,
        'trend_weight':  0.8,
    },
    '30s': {
        'style_boost':  ['Smart Casual', 'Formal', 'Casual'],
        'color_boost':  ['Neutral', 'Classic'],
        'formality_adj':  0.0,
        'trend_weight':   0.5,
    },
    '40s': {
        'style_boost':  ['Formal', 'Smart Casual', 'Classic'],
        'color_boost':  ['Neutral', 'Muted'],
        'formality_adj':  0.1,
        'trend_weight':   0.3,
    },
    '50s+': {
        'style_boost':  ['Formal', 'Minimalist'],
        'color_boost':  ['Neutral', 'Muted'],
        'formality_adj':  0.15,
        'trend_weight':   0.2,
    },
}

# STYLE PREFERENCE → articleType boosted categories
STYLE_TYPE_MAP = {
    'Formal':       ['formal-shirts', 'trousers', 'chinos', 'blazers', 'suits', 'formal-shoes', 'loafers'],
    'Smart Casual': ['casual-shirts', 'linen-shirts', 'polo-tshirts', 'chinos', 'jeans', 'loafers', 'watches'],
    'Casual':       ['tshirts', 'jeans', 'sneakers', 'shorts', 'caps', 'handbags'],
    'Streetwear':   ['tshirts', 'sweatshirts', 'track-pants', 'sneakers', 'caps'],
    'Bohemian':     ['casual-dresses', 'maxi-dresses', 'skirts', 'sandals', 'flats', 'earrings', 'necklaces'],
    'Minimalist':   ['formal-shirts', 'trousers', 'chinos', 'sneakers', 'formal-shoes', 'watches'],
}

# GENDER → dataset filter values (using exact categories in products.csv: 'men', 'women', 'unisex')
GENDER_FILTER_MAP = {
    'Men':    ['men', 'unisex'],
    'Women':  ['women', 'unisex'],
    'Boys':   ['men', 'unisex'],
    'Girls':  ['women', 'unisex'],
    'Unisex': ['men', 'women', 'unisex'],
}

print("✅ Context mapping tables loaded successfully!")"""))


# 7c: Profile-Based Candidate Filtering
cells.append(nbf.v4.new_markdown_cell("""## 7c: Profile-Based Candidate Filtering
Step to filter out completely incompatible products based on gender and avoid tags."""))

cells.append(nbf.v4.new_code_cell("""def filter_by_profile(df: pd.DataFrame, profile: UserProfile) -> pd.Index:
    \"\"\"
    Applies gender filter and occasion avoid_types filter.
    Never returns an empty index (falls back to full df.index if over-filtered).
    \"\"\"
    mask = pd.Series([True] * len(df), index=df.index)
    
    # 1. Gender check
    if profile.gender:
        allowed_genders = GENDER_FILTER_MAP.get(profile.gender, [profile.gender.lower()])
        gender_col = [c for c in df.columns if 'gender' in c.lower()][0]
        # Match case-insensitively
        mask &= df[gender_col].str.lower().isin([g.lower() for g in allowed_genders])
        
    # 2. Occasion check: exclude avoid_types
    if profile.occasion and profile.occasion in OCCASION_PROFILE:
        avoid = OCCASION_PROFILE[profile.occasion]['avoid_types']
        type_col = [c for c in df.columns if 'category' in c.lower() or 'article' in c.lower() or 'type' in c.lower()][0]
        mask &= ~df[type_col].str.lower().isin([a.lower() for a in avoid])
        
    filtered = df[mask].index
    if len(filtered) == 0:
        print(f"⚠️ Over-filtered for profile {profile.summary()}, reverting to full dataset.")
        return df.index
        
    return filtered

# Quick checks
test_filtered = filter_by_profile(df, profile_office_man)
print(f"Office Man Profile: filtered {len(df)} items down to {len(test_filtered)}")
test_filtered_party = filter_by_profile(df, profile_party_woman)
print(f"Party Woman Profile: filtered {len(df)} items down to {len(test_filtered_party)}")"""))


# 7d: Context-Aware Scoring Function
cells.append(nbf.v4.new_markdown_cell("""## 7d: Context-Aware Scoring Function
Upgraded scorer incorporating: compatibility (30%), occasion (25%), color (20%), style preference (15%), and age formality (10%)."""))

cells.append(nbf.v4.new_code_cell("""def context_aware_score(
    seed_idx: int,
    candidate_idx: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    profile: UserProfile
) -> tuple:
    \"\"\"
    Calculates a multi-signal context-aware similarity score in [0.0, 1.0].
    \"\"\"
    # Signal 1: Base compatibility (30%)
    base_score, base_reason = compatibility_score(seed_idx, candidate_idx, df, sim_matrix)
    
    candidate = df.iloc[candidate_idx]
    type_col = [c for c in df.columns if 'category' in c.lower() or 'article' in c.lower() or 'type' in c.lower()][0]
    color_col = [c for c in df.columns if 'color' in c.lower() or 'colour' in c.lower()][0]
    
    item_type = str(candidate.get(type_col, '')).lower()
    item_color = str(candidate.get(color_col, ''))
    
    # Signal 2: Occasion relevance (25%) using Zero-Shot CLIP text-to-image similarity
    occasion_score = 0.5
    if profile.occasion:
        occ_desc_map = {
            'Office': 'business formal office wear',
            'Party': 'glamorous party night wear',
            'Wedding': 'traditional wedding ceremonial wear',
            'Beach': 'casual beach vacation wear',
            'Casual': 'relaxed everyday casual wear',
            'Formal': 'classic formal evening wear',
            'Date Night': 'romantic evening date night wear',
            'Sport': 'athletic sports gym activewear'
        }
        occ_desc = occ_desc_map.get(profile.occasion, f"clothing for {profile.occasion}")
        
        global OCCASION_EMBEDDINGS
        if 'OCCASION_EMBEDDINGS' not in globals():
            OCCASION_EMBEDDINGS = {}
            
        if profile.occasion not in OCCASION_EMBEDDINGS:
            try:
                inputs = processor(text=[occ_desc], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
                with torch.no_grad():
                    text_emb = model.get_text_features(**inputs).cpu().numpy()
                text_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-10)
                OCCASION_EMBEDDINGS[profile.occasion] = text_norm.squeeze()
            except Exception as e:
                pass
                
        if profile.occasion in OCCASION_EMBEDDINGS:
            occ_emb = OCCASION_EMBEDDINGS[profile.occasion]
            cand_vis_norm = visual_norm[candidate_idx]
            sim = float(np.dot(cand_vis_norm, occ_emb))
            # Rescale similarity from typical range [0.15, 0.35] to [0.0, 1.0]
            occasion_score = float(np.clip((sim - 0.18) / (0.32 - 0.18), 0.0, 1.0))
            
    # Signal 3: Color context match (20%)
    color_score = 0.5
    if profile.occasion and profile.occasion in OCCASION_PROFILE:
        pref_colors = [c.lower() for c in OCCASION_PROFILE[profile.occasion]['preferred_colors']]
        neutral_colors = [c.lower() for c in NEUTRAL_COLORS]
        if item_color.lower() in pref_colors:
            color_score = 1.0
        elif item_color.lower() in neutral_colors:
            color_score = 0.7
        else:
            color_score = 0.3
            
    # Signal 4: Style alignment (15%)
    style_score = 0.5
    if profile.style_pref and profile.style_pref in STYLE_TYPE_MAP:
        pref_types = [t.lower() for t in STYLE_TYPE_MAP[profile.style_pref]]
        if item_type in pref_types:
            style_score = 1.0
        else:
            style_score = 0.3
            
    # Signal 5: Age formality fit (10%)
    age_score = 0.5
    if profile.age_group and profile.age_group in AGE_STYLE_WEIGHTS:
        age_cfg = AGE_STYLE_WEIGHTS[profile.age_group]
        target_formality = 0.5 + age_cfg['formality_adj']
        
        # Simple heuristic formality mapping
        formal_types = [t.lower() for t in STYLE_TYPE_MAP.get('Formal', [])]
        casual_types = [t.lower() for t in STYLE_TYPE_MAP.get('Casual', [])]
        
        if item_type in formal_types:
            item_formality = 0.9
        elif item_type in casual_types:
            item_formality = 0.2
        else:
            item_formality = 0.5
            
        age_score = max(0.0, 1.0 - abs(item_formality - target_formality))
        
    final_score = (
        0.30 * base_score +
        0.25 * occasion_score +
        0.20 * color_score +
        0.15 * style_score +
        0.10 * age_score
    )
    
    breakdown = {
        'base': round(base_score, 3),
        'occasion': round(occasion_score, 3),
        'color': round(color_score, 3),
        'style': round(style_score, 3),
        'age_formality': round(age_score, 3),
        'final': round(final_score, 3)
    }
    
    reasons = [base_reason] if base_reason else []
    if occasion_score >= 0.8 and profile.occasion:
        reasons.append(f"suited for {profile.occasion}")
    if color_score >= 0.8 and profile.occasion:
        reasons.append(f"color {item_color} is highly suited for {profile.occasion}")
    if style_score >= 0.8 and profile.style_pref:
        reasons.append(f"matches {profile.style_pref} style preference")
        
    explanation = "; ".join(reasons) if reasons else "Profile-matched style."
    return round(final_score, 3), breakdown, explanation

# Quick test print
t_score, t_bd, t_ex = context_aware_score(0, 1, df, sim_matrix, profile_office_man)
print("Test Score:", t_score)
print("Breakdown:", t_bd)
print("Reason:", t_ex)"""))


# 7e: Profile-Aware Outfit Generator
cells.append(nbf.v4.new_markdown_cell("""## 7e: Profile-Aware Outfit Generator
Generates outfit coordinates utilizing context-aware scoring and hard profile candidate filtering."""))

cells.append(nbf.v4.new_code_cell("""def generate_profile_outfit(
    seed_idx: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    profile: UserProfile,
    top_k: int = 5
) -> dict:
    \"\"\"
    Generates a full context-aware outfit.
    \"\"\"
    seed = df.iloc[seed_idx]
    valid_indices = filter_by_profile(df, profile)
    filtered_df = df.loc[valid_indices]
    
    outfit = {
        'seed': seed.to_dict(),
        'profile': profile.summary(),
        'items': []
    }
    
    type_col = [c for c in df.columns if 'category' in c.lower() or 'article' in c.lower() or 'type' in c.lower()][0]
    
    for slot, categories in OUTFIT_SLOTS.items():
        # Match candidate types
        slot_candidates = filtered_df[filtered_df[type_col].str.lower().isin([c.lower() for c in categories])]
        # Gap 2: Exclude seed item
        slot_candidates = slot_candidates[slot_candidates.index != seed_idx]
        
        # Fallback if over-filtered
        if slot_candidates.empty:
            slot_candidates = df[df[type_col].str.lower().isin([c.lower() for c in categories])]
            slot_candidates = slot_candidates[slot_candidates.index != seed_idx]
            
        if slot_candidates.empty:
            continue
            
        scored = []
        for idx in slot_candidates.index:
            score, breakdown, reason = context_aware_score(seed_idx, idx, df, sim_matrix, profile)
            scored.append((idx, score, breakdown, reason))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0]
        
        outfit['items'].append({
            'slot': slot,
            'item': df.iloc[best[0]].to_dict(),
            'final_score': best[1],
            'breakdown': best[2],
            'reason': best[3],
            'idx': best[0]
        })
        
    return outfit

# Demo print the 4 profiles
print("Demo execution across all 4 sample profiles:")
for p in [profile_office_man, profile_party_woman, profile_casual_teen, profile_wedding_woman]:
    # Select a valid seed item
    valid_seeds = filter_by_profile(df, p)
    s_idx = valid_seeds[0] if len(valid_seeds) > 0 else df.index[0]
    out = generate_profile_outfit(s_idx, df, sim_matrix, p)
    print(f"\\nProfile: {p.summary()}")
    print(f"  Hero Seed: {out['seed']['brand']} {out['seed']['name']} (Category: {out['seed']['category']})")
    for item in out['items']:
        print(f"  → Slot [{item['slot'].upper()}]: {item['item']['brand']} {item['item']['name']} (Score: {item['final_score']:.2f})")
        print(f"    Reason: {item['reason']}")"""))


# 7f: Profile-Aware Chat Integration
cells.append(nbf.v4.new_markdown_cell("""## 7f: Profile-Aware Chat Integration
Helpers to extract demographic details from messages and execute response building."""))

cells.append(nbf.v4.new_code_cell("""# 7f-i: Profile Extractor from Natural Language
def extract_profile_from_text(text: str) -> UserProfile:
    \"\"\"
    Parses natural language queries to extract profiling parameters.
    \"\"\"
    text_lower = text.lower()
    
    # 1. Gender
    gender = None
    gender_map = {
        'Men': ['men', 'male', 'man', 'guy', 'him'],
        'Women': ['women', 'female', 'woman', 'girl', 'her'],
        'Boys': ['boy', 'boys', 'teen boy'],
        'Girls': ['girl', 'girls', 'teen girl'],
    }
    for g, kws in gender_map.items():
        if any(kw in text_lower for kw in kws):
            gender = g
            break
            
    # 2. Age group
    age_group = None
    age_map = {
        '20s': ['20s', 'twenties', '20 year', '25 year', 'college', 'young adult'],
        '30s': ['30s', 'thirties', '30 year', '35 year'],
        '40s': ['40s', 'forties', '40 year', '45 year'],
        '50s+': ['50s', 'fifties', '50 year', '60s', 'senior', 'mature'],
    }
    for ag, kws in age_map.items():
        if any(kw in text_lower for kw in kws):
            age_group = ag
            break
            
    # 3. Occasion
    occasion = None
    occasion_map = {
        'Office': ['office', 'work', 'meeting', 'corporate', 'professional', 'interview', 'business'],
        'Party': ['party', 'club', 'night out', 'cocktail', 'celebration'],
        'Wedding': ['wedding', 'marriage', 'reception', 'sangeet', 'mehendi'],
        'Beach': ['beach', 'vacation', 'holiday', 'summer', 'resort'],
        'Casual': ['casual', 'everyday', 'weekend', 'errands', 'hangout'],
        'Formal': ['formal', 'gala', 'ceremony', 'black tie'],
        'Date Night': ['date', 'dinner', 'romantic', 'evening out'],
        'Sport': ['gym', 'workout', 'sports', 'running', 'exercise', 'training'],
    }
    for occ, kws in occasion_map.items():
        if any(kw in text_lower for kw in kws):
            occasion = occ
            break
            
    # 4. Style Preference
    style_pref = None
    style_map = {
        'Formal': ['formal', 'classic', 'traditional', 'professional'],
        'Smart Casual': ['smart casual', 'smart-casual', 'business casual'],
        'Casual': ['casual', 'relaxed', 'laid back'],
        'Streetwear': ['streetwear', 'street style', 'urban', 'hypebeast'],
        'Bohemian': ['boho', 'bohemian', 'flowy'],
        'Minimalist': ['minimal', 'minimalist', 'clean look'],
    }
    for sp, kws in style_map.items():
        if any(kw in text_lower for kw in kws):
            style_pref = sp
            break
            
    return UserProfile(gender=gender, age_group=age_group, occasion=occasion, style_pref=style_pref)

# 7f-ii: Updated build_response_with_profile()
def build_response_with_profile(
    user_text: str,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    base_profile: Optional[UserProfile] = None
) -> tuple:
    \"\"\"
    Complete response pipeline with UserProfile extraction and overriding.
    \"\"\"
    text_profile = extract_profile_from_text(user_text)
    
    # Merge dropdown profile with text-extracted profile (text overrides dropdown)
    merged = UserProfile(
        gender=text_profile.gender or (base_profile.gender if base_profile else None),
        age_group=text_profile.age_group or (base_profile.age_group if base_profile else None),
        occasion=text_profile.occasion or (base_profile.occasion if base_profile else None),
        style_pref=text_profile.style_pref or (base_profile.style_pref if base_profile else None)
    )
    
    intent = parse_intent(user_text, df)
    valid_indices = filter_by_profile(df, merged)
    seed_idx = valid_indices[0] if len(valid_indices) > 0 else df.index[0]
    
    # Find matching seed category
    type_col = [c for c in df.columns if 'category' in c.lower() or 'article' in c.lower() or 'type' in c.lower()][0]
    for cat in df[type_col].unique():
        if cat.lower() in user_text.lower():
            matches = df.loc[valid_indices][df.loc[valid_indices, type_col] == cat]
            if not matches.empty:
                seed_idx = matches.index[0]
                break
                
    if intent['intent'] == 'complete_outfit':
        outfit = generate_profile_outfit(seed_idx, df, sim_matrix, merged)
        text = f"👤 **Active Profile**: {merged.summary()}\\n"
        text += f"🌟 **Multimodal Outfit for**: {outfit['seed']['brand']} {outfit['seed']['name']}\\n\\n"
        imgs = [outfit['seed']['image']]
        
        for item in outfit['items']:
            score_pct = f"{item['final_score']*100:.0f}%"
            text += f"👉 **Slot [{item['slot'].upper()}]**: {item['item']['brand']} {item['item']['name']} (Compatibility: {score_pct})\\n"
            bd = item['breakdown']
            text += f"   📊 Breakdown — Base:{bd['base']:.2f} | Occasion:{bd['occasion']:.2f} | Color:{bd['color']:.2f} | Style:{bd['style']:.2f} | Age:{bd['age_formality']:.2f}\\n"
            text += f"   💬 Verdict: {item['reason']}\\n\\n"
            imgs.append(item['item']['image'])
            
        text += f"🎨 **Stylist Tip**: This curated outfit was optimized using personal style metrics for a polished visual weight."
        return text, imgs, merged
        
    else:
        # Search queries
        candidates = df.loc[valid_indices]
        if intent['filters']['color']:
            color_col = [c for c in df.columns if 'color' in c.lower()][0]
            candidates = candidates[candidates[color_col] == intent['filters']['color']]
            
        if candidates.empty:
            resp = f"👤 **Active Profile**: {merged.summary()}\\n" \
                   f"Sorry, no matching items found for your search filters and active profile. Try broadening your keywords! 🔍"
            return resp, [], merged
            
        resp = f"👤 **Active Profile**: {merged.summary()}\\n" \
               f"🛍️ Found {len(candidates)} products matching your query:\\n"
        for _, row in candidates.head(5).iterrows():
            resp += f"  • **{row['brand']} {row['name']}** (Color: {row['color']}, Category: {row['category']})\\n"
            
        image_col = [c for c in df.columns if 'image' in c.lower()][0]
        imgs = candidates.head(5)[image_col].tolist()
        return resp, imgs, merged"""))


# 7g: Profile-Aware Chat Widget
cells.append(nbf.v4.new_markdown_cell("""## 7g: Profile-Aware Chat Widget
Widget dashboard combining profile selections and simulated demo outputs."""))

cells.append(nbf.v4.new_code_cell("""# Dropdowns setup
gender_dd  = widgets.Dropdown(
    options=['(not set)', 'Men', 'Women', 'Boys', 'Girls', 'Unisex'],
    description='Gender:', layout=widgets.Layout(width='200px'))

age_dd     = widgets.Dropdown(
    options=['(not set)', '20s', '30s', '40s', '50s+'],
    description='Age Group:', layout=widgets.Layout(width='200px'))

occasion_dd= widgets.Dropdown(
    options=['(not set)', 'Office', 'Party', 'Wedding', 'Beach', 
             'Casual', 'Formal', 'Date Night', 'Sport'],
    description='Occasion:', layout=widgets.Layout(width='200px'))

style_dd   = widgets.Dropdown(
    options=['(not set)', 'Formal', 'Smart Casual', 'Casual', 
             'Streetwear', 'Bohemian', 'Minimalist'],
    description='Style:', layout=widgets.Layout(width='200px'))

profile_panel = widgets.VBox([
    widgets.HTML("<b>👤 Set Your Profile (optional — also typeable in chat)</b>"),
    widgets.HBox([gender_dd, age_dd]),
    widgets.HBox([occasion_dd, style_dd]),
])

profile_chat_history = []

def get_base_profile():
    return UserProfile(
        gender     = None if gender_dd.value   == '(not set)' else gender_dd.value,
        age_group  = None if age_dd.value      == '(not set)' else age_dd.value,
        occasion   = None if occasion_dd.value == '(not set)' else occasion_dd.value,
        style_pref = None if style_dd.value    == '(not set)' else style_dd.value,
    )

input_box2   = widgets.Text(
    placeholder='e.g. "Outfit for a beach vacation for women"',
    layout=widgets.Layout(width='65%'))
send_btn2    = widgets.Button(description='Send 💬', button_style='primary')
chat_out2    = widgets.Output()

def on_send_profile(b):
    user_text = input_box2.value.strip()
    if not user_text: return
    input_box2.value = ''
    
    base = get_base_profile()
    response_text, img_paths, used_profile = build_response_with_profile(
        user_text, df, sim_matrix, base
    )
    
    profile_chat_history.append({
        'user': user_text, 
        'response': response_text,
        'images': img_paths,
        'profile': used_profile.summary()
    })
    
    with chat_out2:
        clear_output()
        for turn in profile_chat_history:
            print(f"🧑 You: {turn['user']}")
            print(f"📌 Profile used: {turn['profile']}")
            print(f"🤖 Assistant:\\n{turn['response']}")
            
            # Matplotlib show recommendations
            imgs = turn.get('images', [])
            import os
            # Ensure file exists using try/except
            valid_imgs = []
            for img_p in imgs:
                if img_p and os.path.exists(img_p):
                    valid_imgs.append(img_p)
            if valid_imgs:
                fig, axes = plt.subplots(1, min(4, len(valid_imgs)), figsize=(12, 3))
                if len(valid_imgs) == 1:
                    axes = [axes]
                for ax, p in zip(axes, valid_imgs[:4]):
                    try:
                        img = Image.open(p).convert('RGB')
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
            print("─" * 60)

send_btn2.on_click(on_send_profile)
chat_widget = widgets.VBox([
    widgets.HTML("<h3>👗 Context-Aware Fashion Assistant</h3>"),
    profile_panel,
    widgets.HTML("<hr>"),
    chat_out2,
    widgets.HBox([input_box2, send_btn2])
])
# display(chat_widget)
"""))

cells.append(nbf.v4.new_code_cell("""# ============================================================
# SECTION 7g STATIC DEMO — visible in saved notebook/GitHub
# ============================================================
PROFILE_DEMO_QUERIES = [
    ("I need a formal outfit for a job interview for men in their 30s",
     UserProfile('Men', '30s', 'Office', 'Formal')),
    ("Beach vacation outfit for a young woman",
     UserProfile('Women', '20s', 'Beach', 'Casual')),
    ("Wedding outfit ideas for women in their 40s",
     UserProfile('Women', '40s', 'Wedding', 'Formal')),
    ("Streetwear look for teenage boys",
     UserProfile('Boys', '20s', 'Casual', 'Streetwear')),
    ("Smart casual dinner date outfit for men",
     UserProfile('Men', '30s', 'Date Night', 'Smart Casual')),
]

print("=" * 70)
print("SECTION 7 — CONTEXT-AWARE ASSISTANT STATIC DEMO")
print("=" * 70)

for query, profile in PROFILE_DEMO_QUERIES:
    print(f"\\n🧑 User: {query}")
    extracted = extract_profile_from_text(query)
    print(f"📌 Extracted Profile: {extracted.summary()}")
    response, imgs, used = build_response_with_profile(
        query, df, sim_matrix, profile
    )
    print(f"🤖 Assistant:\\n{response}")
    print("─" * 70)"""))

# Assign cells to notebook and save
nb['cells'] = cells

with open('fashion_recommendation_assistant.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully as fashion_recommendation_assistant.ipynb!")
