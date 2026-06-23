# Dare XAI – Machine Learning & AI Engineer Intern Assignment
## Assignment: AI Fashion Outfit Recommendation System

Welcome to the **Dare XAI Fashion Outfit Recommendation System** dataset! This package contains a curated subset of fashion items and pre-styled outfits designed to evaluate your capabilities in computer vision, recommendation systems, search retrieval, and natural language understanding.

---

## 📁 Dataset Directory Structure

This curated dataset is organized as follows:
```text
NEWDATASET/
├── README.md               # This documentation file
├── curated25.xlsx          # Original styled outfits spreadsheet
├── outfits.csv             # Cleaned outfit mapping of the 25 curated outfits
├── products.csv            # Detailed metadata for the 68 unique products used in the outfits
└── images/                 # Product image files matching the product IDs
    ├── ajio/               # Images sourced from Ajio
    ├── myntra/             # Images sourced from Myntra
    └── nykaa/              # Images sourced from Nykaa
```

---

## 📊 Data Files Description

### 1. `products.csv`
This file contains the core metadata for the 68 unique fashion items that make up our outfits.
* **Fields**:
  * `id`: Unique identifier for the product (e.g., `ajio_703182002`).
  * `name`: Product title (e.g., `Women Bodycon Midi Length Dress`).
  * `brand`: Manufacturer or label (e.g., `Fyre Rose`, `Peter England`).
  * `price_inr`: Retail price in Indian Rupees (INR).
  * `rating` / `rating_count`: Customer rating statistics.
  * `gender`: Target gender (`men` / `women`).
  * `wear_type`: Style category (e.g., `western`, `ethnic`).
  * `category` & `category_label`: Specific clothing/accessory category (e.g., `formal-shirts`, `heels`, `dresses`).
  * `occasion`: Intended setting (e.g., `party`, `office`, `casual`).
  * `tags`: Semicolon-separated tags for retrieval.
  * `description`: Detailed text description of the product.
  * `image`: Relative filepath to the product image (e.g., `images/ajio/703182002.jpg`).

### 2. `outfits.csv` (and `curated25.xlsx`)
This file defines 25 expert-curated complete outfits. You can use this file as ground truth for training, evaluation, or as reference combinations.
* **Fields**:
  * `outfit_id`: Unique identifier for the outfit (e.g., `outfit W1`).
  * `gender` / `wear_type` / `occasion` / `theme`: Categorization context.
  * `hero` & `hero_id`: The main item in the outfit (e.g., a dress or shirt).
  * `second` & `second_id`: The complementary item (e.g., trousers/chinos).
  * `layer` & `layer_id`: Optional layering item (e.g., blazers, jackets).
  * `footwear` & `footwear_id`: Footwear item.
  * `accessory_1` & `accessory_1_id` / `accessory_2` & `accessory_2_id`: Optional styling accessories.
  * `palette`: Main color combination.
  * `stylist_rationale`: Stylist commentary explaining why this outfit is compatible and fits the theme.

---

## 🎯 Assignment & Problem Statement

Your objective is to design and build an intelligent **Recommendation Engine & Chat-based Fashion Assistant** that can understand natural language user requests, retrieve compatible clothing items, compile complete outfits, and explain its reasoning.

### Core Implementation Requirements:

1. **Dataset Analysis & Understanding**:
   * Inspect the provided metadata and images.
   * Document categories, palette distributions, and potential challenges (e.g., size of dataset, variety, metadata consistency).

2. **Outfit Compatibility Engine**:
   * Build an algorithm to determine if items are compatible. Given a single item (e.g., a white formal shirt), the engine should suggest compatible components (e.g., navy trousers and brown loafers).
   * **Tip**: Use similarity search or learn a pairwise compatibility score using visual/text features.

3. **User & Context-Aware Recommendations**:
   * Adapt recommendations dynamically based on profile parameters:
     * **Gender** (e.g., Men / Women)
     * **Age Group** (e.g., 20s vs. 40s styling)
     * **Occasion** (e.g., Office, Beach Vacation, Wedding, Party)
     * **Style Preferences** (e.g., Formal, Smart Casual)

4. **Conversational Fashion Assistant (Natural Language Interface)**:
   * Build a chat interface allowing users to make requests in plain text (e.g., *"I need an outfit for a business meeting"* or *"Suggest something stylish for a summer beach vacation"*).
   * The assistant should retrieve the items, group them into a complete outfit (Topwear, Bottomwear, Footwear, and optional Accessories/Layers), and display them to the user.

5. **Explainability**:
   * Every outfit recommendation must include a reasoned explanation (e.g., *"Beige chinos pair well with a navy blazer because they provide classic contrast while maintaining a polished smart-casual appearance for your office meeting."*).

---

## 🛠️ Recommended Technical Approach

We evaluate technical depth and systemic engineering choices. Consider incorporating:
* **Computer Vision & Multi-modal Embeddings**: Use models like **CLIP**, **FashionCLIP**, or **SigLIP** to generate embeddings from both product images and descriptions.
* **Vector Search / Retrieval**: Store product embeddings in a vector database (e.g., **Qdrant**, **Chroma**, **FAISS**) to execute fast similarity and hybrid searches.
* **LLM Integration**: Use LLMs (e.g., Gemini, GPT, Claude) to parse user intent from conversational chat, structure queries, and generate final personalized reasoning.
* **Advanced Methods (Bonus)**: Graph-based recommendations (representing outfits as nodes/edges) or trained compatibility classification models.

---

## 🚀 Quick Start Code (Python)

You can load and start exploring this dataset using the following snippet:

```python
import pandas as pd
import os

# Set paths
DATASET_DIR = "./"  # Update path if run from elsewhere
products_df = pd.read_csv(os.path.join(DATASET_DIR, "products.csv"))
outfits_df = pd.read_csv(os.path.join(DATASET_DIR, "outfits.csv"))

print(f"Loaded {len(products_df)} products.")
print(f"Loaded {len(outfits_df)} curated outfits.")

# Example: Display first outfit
first_outfit = outfits_df.iloc[0]
print(f"\nOutfit ID: {first_outfit['outfit_id']} ({first_outfit['theme']})")
print(f"Hero Item: {first_outfit['hero']} (ID: {first_outfit['hero_id']})")
print(f"Footwear: {first_outfit['footwear']} (ID: {first_outfit['footwear_id']})")
print(f"Rationale: {first_outfit['stylist_rationale']}")
```

---
*Good luck with the assignment! We look forward to seeing your creative and technical solutions.*

---

## 🏛️ System Architecture

The Antigravity AI Fashion Stylist system is built on a hybrid architecture that integrates **vector search**, **semantic NLU parsing**, **context-aware rule engines**, and **LLM generation**.

Here is the overall workflow of the conversational stylist:

```mermaid
graph TD
    User([User Prompt]) --> Streamlit[Streamlit Chat Interface]
    Streamlit --> GeminiParser[Gemini NLU Parser]
    GeminiParser -->|Intent, Filters, Profile| Recommender[Local Recommendation Engine]
    
    subgraph Recommender Engine
        Recommender --> ZeroShotSearch[Zero-Shot Semantic Search]
        ZeroShotSearch -->|Query text vs Image Embeddings| CLIPVisual[FashionCLIP Visual Norm]
        
        Recommender --> OutfitGen[Outfit Coordinate Generator]
        OutfitGen --> CategoryFilter[Hard Profile Candidate Filtering]
        OutfitGen --> ScoreScorer[5-Signal Re-Ranking Scorer]
        
        subgraph 5-Signal Scorer
            ScoreScorer --> BaseCompat[Base Styling Compatibility 30%]
            ScoreScorer --> OccasionCLIP[Zero-Shot CLIP Occasion Relevance 25%]
            ScoreScorer --> ColorPalette[Color Context Fit 20%]
            ScoreScorer --> StylePref[Style Alignment Boost 15%]
            ScoreScorer --> AgeGroup[Age Formality Fit 10%]
        end
    end
    
    Recommender -->|Recommended Items + Rationales| GeminiHumanizer[Gemini Stylist Explainer]
    GeminiHumanizer -->|Humanized Stylist Explanation| Streamlit
    Streamlit -->|Rendered Product Cards| User
```

### 1. Outfit Compatibility Engine
The compatibility engine evaluates the styling cohesion between fashion products using a composite score:
$$\text{Compatibility Score} = 0.4 \times \text{Category Match} + 0.3 \times \text{Color Harmony} + 0.3 \times \text{Embedding Similarity}$$

* **Category Compatibility**: Map-based rules (`COMPATIBLE_CATEGORIES`) defining which items pair together structurally (e.g. `formal-shirts` match with `trousers` and `formal-shoes`).
* **Color Harmony**: Checks if color combinations match neutral colors (White, Black, Navy Blue, Beige, Cream, Off White, Grey) or belong to curated harmonious pairs (e.g. Red + Black, Green + Beige, Gold + Black).
* **Multimodal Embedding Similarity**: Calculates cosine similarity using the **FashionCLIP** (`patrickjohncyh/fashion-clip`) average hybrid embeddings of the products:
  $$\text{Embedding Similarity} = \frac{\cos(\mathbf{e}_1, \mathbf{e}_2) + 1}{2}$$
  where $\mathbf{e}_i$ is the hybrid normalized text + vision embedding vector for product $i$.

---

### 2. User & Context-Aware Recommendations
To provide personalized recommendations matching user situations, the engine re-ranks candidate products using 4 profile parameters (Gender, Age Group, Occasion, Style Preference) across 5 scoring signals:

1. **Base Styling Compatibility (30% weight)**: Calculated via the Outfit Compatibility Engine score.
2. **Occasion Relevance (25% weight)**: Replaces rule-based category list mappings with a **zero-shot text-to-image similarity check**. It encodes descriptive occasion text targets (e.g., `"business formal office wear"` for `Office`) via FashionCLIP's text encoder and computes the cosine similarity against the candidate product's image embedding (`visual_norm`).
3. **Color Context (20% weight)**: Scores color alignment depending on the occasion profile (e.g. neutral colors for office, bright/bold colors for parties).
4. **Style Preference (15% weight)**: Boosts product types aligned with the user's specific preference (e.g. Streetwear boosts sweatshirts, sneakers, and track pants).
5. **Age Formality Fit (10% weight)**: Measures the formality deviation between age-group styling targets (e.g. younger 20s prefer relaxed formality; mature 50s+ prefer structured formality) and candidate product category formalities.

---

## 📊 Dataset Analysis & EDA Summary

A comprehensive exploratory data analysis of the cloned dataset was performed in the notebook:
* **Total Products**: 68 unique products.
* **Curated Outfits**: 25 expert-curated outfits (ground-truth reference pairs).
* **Category Distribution**: 47 unique categories (e.g. `formal-shirts`, `party-dresses`, `jeans`, `running-shoes`), indicating a high degree of class variety but significant class imbalance (many categories contain only 1 item).
* **Color Palette Distribution**: Main colors were extracted from text descriptions, with Red (9), White (8), Black (8), Navy Blue (8), Brown (8), and Grey (4) making up the top colors.
* **Image Resolutions**:
  * Widths range from 256px to 420px (Mean: 370.8px).
  * Heights range from 341px to 560px (Mean: 494.3px).
* **Dataset Challenges Addressed**:
  * **AVIF Images**: 28 of the 68 product images are encoded as AVIF format but saved with `.jpg` extensions. Standard Pillow (`PIL.Image`) crashes on these without the `pillow-avif-plugin` installed and explicitly imported in the notebook.
  * **Metadata Gaps**: The original `products.csv` lacks an explicit "color" column. We successfully extracted colors by matching names and descriptions against a strict target color list.

---

## 📓 What's Inside the Notebook (`fashion_recommendation_assistant.ipynb`)

The notebook contains 8 sections, structured as follows:

* **## SECTION 0: Setup & Install**
  * Core imports and dependency installations.
  * Explicit loading of the `pillow-avif-plugin` to register the AVIF decoder.
  * Verification of the T4 GPU runtime.
* **## SECTION 1: Dataset Analysis & EDA**
  * Automated repository cloning and path setup.
  * Core pandas analysis (shapes, null counts, unique value prints).
  * Seaborn distribution plots (category count, color count, and gender/occasion count).
  * Matplotlib 5x5 product image grid with text overlay (robustly wrapped in try/except).
  * Markdown summary documenting image statistics.
* **## SECTION 2: Feature Extraction Pipeline**
  * **2a (Text)**: Encodes concatenated metadata using the pretrained **FashionCLIP** text encoder into 512-dimensional text embeddings.
  * **2b (Visual)**: Extracts image features using the **FashionCLIP** vision encoder into 512-dimensional visual embeddings.
  * **2c (Multimodal Hybrid Indexing)**: Normalizes both modalities, averages them to create a hybrid multimodal representation, and builds a FAISS `IndexFlatIP` (Inner Product) index.
* **## SECTION 3: Outfit Compatibility Engine**
  * **3a**: A dictionary mapping each article category to its compatible categories.
  * **3b**: Color harmony logic handling neutral pairings and classic style matches.
  * **3c**: Compatibility Scorer computing: `Score = 0.4*category_match + 0.3*color_harmony + 0.3*embedding_similarity` and returning natural language explanations.
  * **3d**: Outfit Generator that builds coord-outfits (top, bottom, footwear, accessories) around any seed item.
* **## SECTION 4: Chat-based Fashion Assistant**
  * **4a**: Keyword-based intent parser detecting search, outfit generation, compatibility checks, and style advice, preserving the original query text.
  * **4b**: Conversational response builder leveraging **FashionCLIP** to execute zero-shot text-to-image semantic search by ranking image embeddings (`visual_norm`) against the query text embedding.
  * **4c**: Interactive `ipywidgets` chat widget with styled bubbles and a print-based `chat_simulation()` fallback.
* **## SECTION 5: Evaluation & Visualization**
  * **5a (Quantitative Evaluation)**: Validates the engine against the 25 curated outfits in `outfits.csv`.
    * *Curated Outfits Score*: **0.785 ± 0.052** (with 100% scoring above `0.7`).
    * *Random control baseline*: **0.448**.
    * Plots a density comparison histogram.
  * **5b (Visual Showcase)**: Renders a 1x4 horizontal image grid showing 5 diverse seeds and their recommended outfit items.
  * **5c (PCA Cluster Visualization)**: Projects hybrid embeddings to 2D using PCA, color-coded by master category (Apparel, Footwear, Accessories), annotated with typical categories.
* **## SECTION 6: Demo Showcase Cell**
  * Demonstrates three end-to-end examples with output: text search queries, outfit generation from a seed item, and a simulated 5-turn conversation transcript.
* **## SECTION 7: User & Context-Aware Recommendations**
  * **7a (User Profile)**: Defines the `UserProfile` dataclass mapping gender, age group, occasion context, and style preferences.
  * **7b (Context Mappings)**: Implements static rules for preferred styles, preferred colors, and avoided types per occasion and age segment.
  * **7c (Candidate Filtering)**: Eliminates gender-incompatible products and excluded items per occasion context.
  * **7d (Context Scorer)**: Re-ranks products using a 5-signal soft scoring formula: Base Similarity (30%), Occasion Relevance (25% calculated as the zero-shot cosine similarity between candidate visual embeddings and target occasion text descriptions encoded by FashionCLIP), Color Context (20%), Style Alignment (15%), and Age Formality (10%).
  * **7e (Profile Outfit Generator)**: Compiles outfits matching slots based on context-aware scores.
  * **7f (Profile Chat)**: Parses natural language queries to extract user profiles and overrides dropdown settings.
  * **7g (Profile Chat Dashboard & Demo)**: Builds an interactive widget dashboard with dropdown selectors and executes a static demo printing outcomes for 5 diverse test queries with detailed signal score breakdowns.


