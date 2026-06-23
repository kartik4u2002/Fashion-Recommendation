import streamlit as st
import os
import base64
from PIL import Image
import pillow_avif
import json

# Load environment variables from .env file manually if it exists
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

from recommender import (
    df, UserProfile, generate_profile_outfit, zero_shot_search, 
    compatibility_score, get_clip_model
)

# Page configuration
st.set_page_config(
    page_title="AI Fashion Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local image helper
def get_image_base64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return ""

# Helper function to render product cards
def render_grid(items):
    cards_html = []
    for item_data in items:
        slot = item_data.get('slot', 'item').upper()
        item = item_data.get('item', item_data)
        brand = item.get('brand', 'Unknown')
        name = item.get('name', 'Unknown')
        price = item.get('price_inr', item.get('price', 0))
        reason = item_data.get('reason', item_data.get('rationale', ''))
        score = int(item_data.get('score', item_data.get('final_score', 0.9)) * 100)
        img_path = item.get('image', '')
        
        # Base64 encode image
        img_base64 = ""
        if img_path and os.path.exists(img_path):
            try:
                img_base64 = get_image_base64(img_path)
            except Exception:
                pass
                
        img_src = f"data:image/jpeg;base64,{img_base64}" if img_base64 else "https://via.placeholder.com/260x320?text=No+Image"
        
        card_html = (
            f'<div class="fashion-card">'
            f'<div class="badge">{slot} • {score}% MATCH</div>'
            f'<img src="{img_src}" />'
            f'<div class="info">'
            f'<div class="brand">{brand}</div>'
            f'<div class="name">{name}</div>'
            f'<div class="price">₹{price:,}</div>'
            f'<div class="rationale">💬 {reason}</div>'
            f'</div>'
            f'</div>'
        )
        cards_html.append(card_html)
        
    grid_html = f'<div class="fashion-grid">{"".join(cards_html)}</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

# Premium custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

/* Custom Page Header Gradient */
.header-container {
    background: linear-gradient(135deg, #1e1e2f 0%, #3a3f58 100%);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
}

.header-container h1 {
    font-weight: 700;
    margin: 0;
    font-size: 38px;
    letter-spacing: -0.5px;
}

.header-container p {
    font-size: 16px;
    opacity: 0.8;
    margin-top: 10px;
    margin-bottom: 0;
}

/* Styling fashion recommendation cards */
.fashion-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    justify-content: flex-start;
    margin-top: 20px;
    margin-bottom: 20px;
}

.fashion-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    width: 250px;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    border: 1px solid #eaeaea;
    display: flex;
    flex-direction: column;
    position: relative;
}

.fashion-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
    border-color: #007bff;
}

.fashion-card img {
    width: 100%;
    height: 300px;
    object-fit: cover;
}

.fashion-card .badge {
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(0, 123, 255, 0.9);
    color: white;
    padding: 4px 10px;
    font-size: 10px;
    font-weight: bold;
    border-radius: 20px;
    backdrop-filter: blur(4px);
    z-index: 10;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.fashion-card .info {
    padding: 15px;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}

.fashion-card .brand {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    font-weight: 600;
}

.fashion-card .name {
    font-size: 14px;
    font-weight: 600;
    color: #111;
    margin-top: 4px;
    margin-bottom: 8px;
    height: 38px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.fashion-card .price {
    font-size: 17px;
    font-weight: 700;
    color: #222;
    margin-bottom: 8px;
}

.fashion-card .rationale {
    font-size: 11px;
    color: #555;
    line-height: 1.4;
    border-top: 1px solid #f0f0f0;
    padding-top: 8px;
    margin-top: auto;
}

/* Sidebar Styling Overrides */
[data-testid="stSidebar"] {
    background-color: #f7f9fc;
}

/* Force dark text color for visibility of labels and context overrides on light sidebar */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
    color: #1e1e2f !important;
}

.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: #1e1e2f;
    margin-bottom: 15px;
    border-bottom: 2px solid #eaeaea;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="header-container">
    <h1>✨ AI Fashion Stylist</h1>
    <p>Conversational outfit recommendation assistant powered by FashionCLIP & Gemini</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown('<div class="sidebar-title">⚙️ Stylist Dashboard</div>', unsafe_allow_html=True)

# API Key configuration
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Free API key from Google AI Studio")
api_key = gemini_key or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.sidebar.warning("🔑 Please enter a Gemini API Key to chat with the stylist.")

# Profile Manual Dropdown overrides
st.sidebar.markdown('**👤 Context Overrides (Optional)**')
sb_gender = st.sidebar.selectbox("Target Gender", ['(not set)', 'Men', 'Women', 'Boys', 'Girls', 'Unisex'])
sb_age = st.sidebar.selectbox("Age Group", ['(not set)', '20s', '30s', '40s', '50s+'])
sb_occasion = st.sidebar.selectbox("Occasion Type", ['(not set)', 'Office', 'Party', 'Wedding', 'Beach', 'Casual', 'Formal', 'Date Night', 'Sport'])
sb_style = st.sidebar.selectbox("Style Preference", ['(not set)', 'Formal', 'Smart Casual', 'Casual', 'Streetwear', 'Bohemian', 'Minimalist'])

# Clear conversation button
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "items" in msg and msg["items"]:
            # Display items in visual grid
            render_grid(msg["items"])

# Product display cards render successfully using helper defined above

# Parse user query using Gemini API
def parse_query_with_gemini(user_message: str, api_key: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an AI NLU parser for a fashion recommendation engine.
    Analyze the user request and extract intent, demographic filters, and style context.
    
    User Request: "{user_message}"
    
    Available Occasions: ["Office", "Party", "Wedding", "Beach", "Casual", "Formal", "Date Night", "Sport"]
    Available Genders: ["Men", "Women", "Boys", "Girls", "Unisex"]
    Available Age Groups: ["20s", "30s", "40s", "50s+"]
    Available Style Preferences: ["Formal", "Smart Casual", "Casual", "Streetwear", "Bohemian", "Minimalist"]
    
    Extract the following fields into a JSON object:
    - "intent": Must be one of ["find_item", "complete_outfit", "compatibility_check", "style_advice"].
      * Choose "complete_outfit" if the user wants an outfit coordinate, outfit pairing, or styling look (e.g., "suggest a beach vacation outfit", "pair something with a blue shirt").
      * Choose "find_item" if they want to browse/find specific products (e.g., "find me a white dress", "show me running shoes").
      * Choose "compatibility_check" if they ask if two items match or go well (e.g., "does a black shirt match khaki pants").
      * Choose "style_advice" if they want general styling tips or advice.
    - "gender": Standardized string from Genders list if mentioned, otherwise null.
    - "age_group": Standardized string from Age Groups list if mentioned, otherwise null.
    - "occasion": Standardized string from Occasions list if mentioned, otherwise null.
    - "style_pref": Standardized string from Style Preferences list if mentioned, otherwise null.
    - "color": A specific color (e.g. "White", "Navy Blue") if mentioned, otherwise null.
    - "category": A specific clothing category name (e.g. "shirts", "jeans", "dresses", "shoes") if mentioned, otherwise null.
    - "parsed_query": A clean search string to use for vector search (e.g., "navy blue formal shirt", "casual summer dress"), or null.
    
    Return ONLY a valid JSON object. Do not include markdown code block formatting (like ```json).
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip("` \n")
        return json.loads(text)
    except Exception as e:
        return {
            "intent": "complete_outfit" if "outfit" in user_message.lower() or "wear" in user_message.lower() else "find_item",
            "gender": None,
            "age_group": None,
            "occasion": None,
            "style_pref": None,
            "color": None,
            "category": None,
            "parsed_query": user_message
        }

# Generate stylistic explanation using Gemini
def generate_stylist_explanation(user_message: str, parsed_context: dict, recommender_output: dict, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    context_str = json.dumps(recommender_output, indent=2)
    
    prompt = f"""
    You are an expert senior fashion stylist and conversational styling assistant.
    The user is asking: "{user_message}"
    
    Our recommendation engine has retrieved the following clothing products and local compatibility rationales:
    {context_str}
    
    Write a highly engaging, warm, and professional stylist response (using emojis) that presents these items as a complete recommended style.
    
    CRITICAL INSTRUCTIONS:
    1. Do NOT reference raw math scores, numbers, or formula names (like "compatibility score", "0.4*category_match", "cosine similarity"). Instead, speak like a human stylist.
    2. Explain in detail the styling logic:
       - Why the color palette is harmonious (e.g. contrast, neutrals, monochromatic).
       - Why the article types are compatible (e.g. balancing fits, structured layering).
       - How the outfit is tailored to the user's occasion context (e.g. office formal, beach casual) and demographic preference.
    3. Keep the tone sophisticated, stylish, and encouraging.
    4. Format the text nicely in markdown. Do not repeat the items list in a plain list if you're explaining them, but walk through the outfit slots (Hero, Bottom, Footwear, Layer, Accessories) gracefully.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"✨ Here is your personalized recommendation! I've curated a beautiful styling look matching your request. The look features a coordinating color palette and structural fits designed to elevate your style. Enjoy your cohesive outfit!"

# Chat inputs handler
if user_query := st.chat_input("Message the fashion assistant... (e.g. 'I need a wedding outfit for women')"):
    # Display user bubble
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    if not api_key:
        st.error("🔑 Please enter a valid Gemini API Key in the sidebar to get recommendations.")
    else:
        with st.spinner("Stylist is thinking... 👚"):
            # 1. Parse user query via Gemini
            gemini_parsed = parse_query_with_gemini(user_query, api_key)
            
            # 2. Merge overrides & text parsing
            gender = None if sb_gender == '(not set)' else sb_gender
            if not gender:
                gender = gemini_parsed.get('gender')
                
            age_group = None if sb_age == '(not set)' else sb_age
            if not age_group:
                age_group = gemini_parsed.get('age_group')
                
            occasion = None if sb_occasion == '(not set)' else sb_occasion
            if not occasion:
                occasion = gemini_parsed.get('occasion')
                
            style_pref = None if sb_style == '(not set)' else sb_style
            if not style_pref:
                style_pref = gemini_parsed.get('style_pref')
                
            profile = UserProfile(gender=gender, age_group=age_group, occasion=occasion, style_pref=style_pref)
            
            intent = gemini_parsed.get('intent', 'complete_outfit')
            parsed_query = gemini_parsed.get('parsed_query') or user_query
            
            # Filters block
            filters = {}
            if gemini_parsed.get('color'):
                filters['color'] = gemini_parsed.get('color')
            if gemini_parsed.get('category'):
                filters['category'] = gemini_parsed.get('category')
            if gender:
                filters['gender'] = gender.lower() if gender.lower() in ['men', 'women'] else gender
                
            recommender_results = {}
            rendered_items = []
            
            # 3. Retrieve from core recommender
            if intent == 'complete_outfit':
                # Rank seeds by query match using zero-shot visual search
                search_results = zero_shot_search(parsed_query, filters)
                if search_results:
                    seed_item = search_results[0]
                    seed_idx = df[df['id'] == seed_item['id']].index[0]
                else:
                    # Fallback to profile filtering
                    valid_idx = filter_by_profile(profile)
                    seed_idx = valid_idx[0] if len(valid_idx) > 0 else 0
                
                # Generate outfit coordinates
                outfit = generate_profile_outfit(seed_idx, profile)
                
                # Wrap results for Gemini
                recommender_results = {
                    "outfit_seed": outfit["seed"],
                    "profile_applied": profile.summary(),
                    "recommended_coordinates": []
                }
                
                rendered_items.append({
                    "slot": "👚 HERO (SEED)",
                    "item": outfit["seed"],
                    "score": 1.0,
                    "reason": "The selected base item matching your request."
                })
                
                for coord in outfit.get("items", []):
                    rendered_items.append({
                        "slot": coord.get("slot", "item"),
                        "item": coord.get("item"),
                        "score": coord.get("final_score", 0.9),
                        "reason": coord.get("reason", "")
                    })
                    recommender_results["recommended_coordinates"].append({
                        "slot": coord.get("slot"),
                        "brand": coord.get("item", {}).get("brand"),
                        "name": coord.get("item", {}).get("name"),
                        "price": coord.get("item", {}).get("price_inr"),
                        "color": coord.get("item", {}).get("color"),
                        "occasion": coord.get("item", {}).get("occasion"),
                        "stylist_score_breakdown": coord.get("breakdown"),
                        "local_pairing_logic": coord.get("reason")
                    })
                    
            elif intent == 'find_item':
                search_results = zero_shot_search(parsed_query, filters)
                recommender_results = {
                    "matched_items": search_results[:3],
                    "profile_applied": profile.summary()
                }
                for rank, res in enumerate(search_results[:3]):
                    rendered_items.append({
                        "slot": f"🛍️ Match {rank+1}",
                        "item": res,
                        "score": res.get("similarity", 0.8),
                        "reason": f"Visual-textual similarity search match for '{parsed_query}'."
                    })
                    
            elif intent == 'compatibility_check':
                # Evaluate compatibility of random or keyword-specified items
                matched_items = zero_shot_search(parsed_query, filters)
                if len(matched_items) >= 2:
                    p1, p2 = matched_items[0], matched_items[1]
                else:
                    p1 = df.sample(1).iloc[0].to_dict()
                    p2 = df.sample(1).iloc[0].to_dict()
                    
                score, explanation = compatibility_score(p1['id'], p2['id'])
                recommender_results = {
                    "item1": p1,
                    "item2": p2,
                    "compatibility_score": score,
                    "local_styling_reason": explanation
                }
                rendered_items.extend([
                    {"slot": "👔 Item A", "item": p1, "score": 1.0, "reason": "First styling candidate."},
                    {"slot": "👖 Item B", "item": p2, "score": score, "reason": explanation}
                ])
                
            else: # style_advice
                recommender_results = {
                    "advice_topics": ["Neutrals balancing", "Fit silhouette coordination", "Footwear occasion anchoring"]
                }
            
            # 4. Generate explanation using Gemini API
            explanation = generate_stylist_explanation(user_query, gemini_parsed, recommender_results, api_key)
            
            # 5. Display response
            with st.chat_message("assistant"):
                st.markdown(explanation)
                if rendered_items:
                    render_grid(rendered_items)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": explanation,
                "items": rendered_items
            })
