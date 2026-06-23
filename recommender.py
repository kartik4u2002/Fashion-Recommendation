import pandas as pd
import numpy as np
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# Path setup
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Helper for color extraction
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

# Load files
products_path = os.path.join(DATA_DIR, 'products.csv')
outfits_path = os.path.join(DATA_DIR, 'outfits.csv')
sim_matrix_path = os.path.join(DATA_DIR, 'sim_matrix.npy')
visual_embeddings_path = os.path.join(DATA_DIR, 'visual_embeddings.npy')

if not os.path.exists(products_path):
    raise FileNotFoundError(f"Missing products.csv in {DATA_DIR}")

df = pd.read_csv(products_path)
df['color'] = df.apply(extract_color, axis=1)

outfits_df = pd.read_csv(outfits_path) if os.path.exists(outfits_path) else pd.DataFrame()

# Load arrays
if os.path.exists(sim_matrix_path):
    sim_matrix = np.load(sim_matrix_path)
else:
    sim_matrix = None

visual_norm = None
if os.path.exists(visual_embeddings_path):
    visual_embeddings = np.load(visual_embeddings_path)
    # L2 normalize
    norms = np.linalg.norm(visual_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    visual_norm = visual_embeddings / norms

# Lazy-load Model & Processor
model = None
processor = None
device = None

def get_clip_model():
    global model, processor, device
    if model is None:
        from transformers import CLIPProcessor, CLIPModel
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Lazy loading patrickjohncyh/fashion-clip on device: {device}")
        model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip').to(device)
        processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')
        model.eval()
    return model, processor, device

# Core Dictionaries
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

for cat in df['category'].unique():
    if cat not in COMPATIBLE_CATEGORIES:
        COMPATIBLE_CATEGORIES[cat] = []

NEUTRAL_COLORS = ["White", "Black", "Grey", "Navy Blue", "Beige", "Off White", "Cream", "Dark Grey"]
COLOR_HARMONY = {
    'neutrals': NEUTRAL_COLORS,
    'pairs': {
        frozenset(["Red", "Black"]), frozenset(["Red", "White"]), frozenset(["Red", "Grey"]),
        frozenset(["Blue", "Beige"]), frozenset(["Blue", "White"]), frozenset(["Green", "Beige"]),
        frozenset(["Green", "White"]), frozenset(["Pink", "Grey"]), frozenset(["Pink", "White"]),
        frozenset(["Yellow", "Navy Blue"]), frozenset(["Yellow", "White"]), frozenset(["Gold", "Black"]),
        frozenset(["Gold", "Red"]), frozenset(["Lavender", "Blue"]), frozenset(["Lavender", "White"]),
        frozenset(["Lavender", "Grey"]), frozenset(["Olive", "Black"]), frozenset(["Olive", "Beige"]),
        frozenset(["Maroon", "Black"]), frozenset(["Maroon", "White"]), frozenset(["Cream", "Gold"]),
        frozenset(["Tan", "Navy Blue"]), frozenset(["Tan", "White"])
    }
}

OUTFIT_SLOTS = {
    'top': ['formal-shirts', 'casual-shirts', 'linen-shirts', 'party-shirts', 'tshirts', 'polo-tshirts', 'sweatshirts', 'sweaters', 'tops'],
    'bottom': ['trousers', 'jeans', 'chinos', 'shorts', 'track-pants', 'skirts', 'leggings'],
    'footwear': ['running-shoes', 'sneakers', 'ethnic-footwear', 'heels', 'boots', 'flats', 'formal-shoes', 'loafers', 'sandals'],
    'layer': ['blazers', 'suits', 'nehru-jackets', 'denim-jackets', 'long-coats'],
    'dress': ['party-dresses', 'casual-dresses', 'maxi-dresses', 'co-ord-sets', 'kurta-sets', 'sherwanis', 'sharara-sets', 'salwar-suits', 'wedding-sarees'],
    'accessory': ['necklaces', 'earrings', 'clutches', 'handbags', 'sunglasses', 'watches', 'caps']
}

SLOT_MAPPING = {article_type: slot_name for slot_name, types in OUTFIT_SLOTS.items() for article_type in types}

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

STYLE_TYPE_MAP = {
    'Formal':       ['formal-shirts', 'trousers', 'chinos', 'blazers', 'suits', 'formal-shoes', 'loafers'],
    'Smart Casual': ['casual-shirts', 'linen-shirts', 'polo-tshirts', 'chinos', 'jeans', 'loafers', 'watches'],
    'Casual':       ['tshirts', 'jeans', 'sneakers', 'shorts', 'caps', 'handbags'],
    'Streetwear':   ['tshirts', 'sweatshirts', 'track-pants', 'sneakers', 'caps'],
    'Bohemian':     ['casual-dresses', 'maxi-dresses', 'skirts', 'sandals', 'flats', 'earrings', 'necklaces'],
    'Minimalist':   ['formal-shirts', 'trousers', 'chinos', 'sneakers', 'formal-shoes', 'watches'],
}

GENDER_FILTER_MAP = {
    'Men':    ['men', 'unisex'],
    'Women':  ['women', 'unisex'],
    'Boys':   ['men', 'unisex'],
    'Girls':  ['women', 'unisex'],
    'Unisex': ['men', 'women', 'unisex'],
}

OCCASION_DESCRIPTIONS = {
    'Office': 'business formal office wear',
    'Party': 'glamorous party night wear',
    'Wedding': 'traditional wedding ceremonial wear',
    'Beach': 'casual beach vacation wear',
    'Casual': 'relaxed everyday casual wear',
    'Formal': 'classic formal evening wear',
    'Date Night': 'romantic evening date night wear',
    'Sport': 'athletic sports gym activewear'
}

OCCASION_EMBEDDINGS = {}

@dataclass
class UserProfile:
    gender:      Optional[str] = None
    age_group:   Optional[str] = None
    occasion:    Optional[str] = None
    style_pref:  Optional[str] = None
    
    def summary(self) -> str:
        parts = []
        if self.gender:    parts.append(f"Gender: {self.gender}")
        if self.age_group: parts.append(f"Age: {self.age_group}")
        if self.occasion:  parts.append(f"Occasion: {self.occasion}")
        if self.style_pref:parts.append(f"Style: {self.style_pref}")
        return " | ".join(parts) if parts else "No profile set"

def color_compatible(c1, c2) -> bool:
    if c1 == 'Unknown' or c2 == 'Unknown':
        return True
    if c1 == c2:
        return True
    if c1 in COLOR_HARMONY['neutrals'] or c2 in COLOR_HARMONY['neutrals']:
        return True
    if frozenset([c1, c2]) in COLOR_HARMONY['pairs']:
        return True
    return False

def compatibility_score(id1_or_idx, id2_or_idx, df=df, sim_matrix=sim_matrix) -> Tuple[float, str]:
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
    
    # Category matching
    cat1, cat2 = row1['category'], row2['category']
    cat_match = 0.0
    if cat2 in COMPATIBLE_CATEGORIES.get(cat1, []) or cat1 in COMPATIBLE_CATEGORIES.get(cat2, []):
        cat_match = 1.0
        
    # Color harmony matching
    col1, col2 = row1['color'], row2['color']
    col_match = 1.0 if color_compatible(col1, col2) else 0.0
    
    # Embedding similarity
    sim = sim_matrix[idx1, idx2] if sim_matrix is not None else 0.5
    sim_score = float(np.clip((sim + 1) / 2, 0.0, 1.0))
    
    total_score = 0.4 * cat_match + 0.3 * col_match + 0.3 * sim_score
    
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
    
    return total_score, explanation

def generate_complete_outfit(seed_id, df=df, sim_matrix=sim_matrix) -> dict:
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
        
        for idx, row in df.iterrows():
            if idx == seed_idx:
                continue
            item_slot = SLOT_MAPPING.get(row['category'], 'accessory')
            if slot == 'top' and item_slot == 'dress':
                pass
            elif item_slot != slot:
                continue
                
            if seed_gender != 'unisex' and row['gender'] != 'unisex' and row['gender'] != seed_gender:
                continue
                
            score, explanation = compatibility_score(seed_id, row['id'], df, sim_matrix)
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
    }

def filter_by_profile(profile: UserProfile, df=df) -> pd.Index:
    mask = pd.Series([True] * len(df), index=df.index)
    
    if profile.gender:
        allowed_genders = GENDER_FILTER_MAP.get(profile.gender, [profile.gender.lower()])
        gender_col = [c for c in df.columns if 'gender' in c.lower()][0]
        mask &= df[gender_col].str.lower().isin([g.lower() for g in allowed_genders])
        
    if profile.occasion and profile.occasion in OCCASION_PROFILE:
        avoid = OCCASION_PROFILE[profile.occasion]['avoid_types']
        type_col = [c for c in df.columns if 'category' in c.lower() or 'article' in c.lower() or 'type' in c.lower()][0]
        mask &= ~df[type_col].str.lower().isin([a.lower() for a in avoid])
        
    filtered = df[mask].index
    if len(filtered) == 0:
        return df.index
    return filtered

def context_aware_score(
    seed_idx: int,
    candidate_idx: int,
    profile: UserProfile,
    df=df,
    sim_matrix=sim_matrix
) -> Tuple[float, dict, str]:
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
        occ_desc = OCCASION_DESCRIPTIONS.get(profile.occasion, f"clothing for {profile.occasion}")
        
        global OCCASION_EMBEDDINGS
        if profile.occasion not in OCCASION_EMBEDDINGS:
            try:
                clip_model, clip_proc, clip_dev = get_clip_model()
                inputs = clip_proc(text=[occ_desc], padding=True, truncation=True, max_length=77, return_tensors="pt").to(clip_dev)
                with torch.no_grad():
                    text_emb = clip_model.get_text_features(**inputs).cpu().numpy()
                text_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-10)
                OCCASION_EMBEDDINGS[profile.occasion] = text_norm.squeeze()
            except Exception as e:
                print(f"Error extracting occasion embedding for {profile.occasion}: {e}")
                
        if profile.occasion in OCCASION_EMBEDDINGS and visual_norm is not None:
            occ_emb = OCCASION_EMBEDDINGS[profile.occasion]
            cand_vis_norm = visual_norm[candidate_idx]
            sim = float(np.dot(cand_vis_norm, occ_emb))
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

def generate_profile_outfit(
    seed_idx: int,
    profile: UserProfile,
    df=df,
    sim_matrix=sim_matrix,
    top_k: int = 5
) -> dict:
    seed = df.iloc[seed_idx]
    valid_indices = filter_by_profile(profile, df)
    filtered_df = df.loc[valid_indices]
    
    outfit = {
        'seed': seed.to_dict(),
        'profile': profile.summary(),
        'items': []
    }
    
    type_col = [c for c in df.columns if 'category' in c.lower() or 'article' in c.lower() or 'type' in c.lower()][0]
    
    for slot, categories in OUTFIT_SLOTS.items():
        slot_candidates = filtered_df[filtered_df[type_col].str.lower().isin([c.lower() for c in categories])]
        slot_candidates = slot_candidates[slot_candidates.index != seed_idx]
        
        if slot_candidates.empty:
            slot_candidates = df[df[type_col].str.lower().isin([c.lower() for c in categories])]
            slot_candidates = slot_candidates[slot_candidates.index != seed_idx]
            
        if slot_candidates.empty:
            continue
            
        scored = []
        for idx in slot_candidates.index:
            score, breakdown, reason = context_aware_score(seed_idx, idx, profile, df, sim_matrix)
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

def zero_shot_search(query_text: str, filters: dict = None, df=df) -> List[dict]:
    if not query_text:
        return []
    
    candidates = df.copy()
    if filters:
        for key, val in filters.items():
            if val:
                candidates = candidates[candidates[key] == val]
                
    try:
        clip_model, clip_proc, clip_dev = get_clip_model()
        inputs = clip_proc(text=[query_text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(clip_dev)
        with torch.no_grad():
            query_emb = clip_model.get_text_features(**inputs).cpu().numpy()
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
        
        if visual_norm is not None:
            similarities = np.dot(visual_norm, query_norm.T).squeeze()
            df_temp = df.copy()
            df_temp['similarity'] = similarities
            
            if not candidates.empty:
                candidates = df_temp.loc[candidates.index].sort_values(by='similarity', ascending=False)
            else:
                candidates = df_temp.sort_values(by='similarity', ascending=False)
                
            results = []
            for idx, row in candidates.iterrows():
                item = row.to_dict()
                item['similarity'] = float(row['similarity'])
                results.append(item)
            return results
    except Exception as e:
        print(f"Error during CLIP zero-shot search: {e}")
        
    # Fallback to random if failed
    return candidates.sample(min(len(candidates), 5)).to_dict('records')
