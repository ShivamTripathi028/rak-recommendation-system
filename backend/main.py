# rak_recommender_local/main.py

# --- Standard Library Imports ---
import os
import pickle
import re
import json
import datetime
from typing import List, Dict, Any, Optional

# --- Third-Party Imports ---
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
except ImportError:
    print("CRITICAL: spaCy not found. Please ensure it's in requirements.txt and installed.")
    raise
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("CRITICAL: sentence-transformers not found. Please ensure it's in requirements.txt and installed.")
    raise

# --- Application Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") 
LOGS_DIR = os.path.join(BASE_DIR, "logs")

SOURCE_EXCEL_FILE = os.path.join(BASE_DIR, "RAKWireless Recommender Dataset.xlsx") 
PRODUCTS_PKL = os.path.join(DATA_DIR, "df_products_enhanced.pkl")
FEATURES_PKL = os.path.join(DATA_DIR, "df_features.pkl")
MAPPING_PKL = os.path.join(DATA_DIR, "df_mapping_exploded.pkl")
EMBEDDINGS_NPY = os.path.join(DATA_DIR, "product_embeddings.npy")

SPACY_MODEL_NAME = "en_core_web_lg"
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
STORE_URL = os.getenv("STORE_URL", "https://store.rakwireless.com/") 

SCORE_HARD_REQUIREMENT = 100
SCORE_STRONG_PREFERENCE = 60
SCORE_SOFT_RELEVANCE = 30
SCORE_SEMANTIC_SIMILARITY_BOOST = 75
DEFAULT_SEMANTIC_THRESHOLD = 0.60

# --- Global Variables ---
df_products_global: Optional[pd.DataFrame] = None
df_features_global: Optional[pd.DataFrame] = None
product_embeddings_global: Optional[np.ndarray] = None
nlp_global: Optional[spacy.language.Language] = None
sbert_model_global: Optional[SentenceTransformer] = None

CONNECTIVITY_KEYWORDS_DICT: Dict[str, str] = {}
# CATEGORY_KEYWORDS_DICT: Dict[str, str] = {} # --- REMOVED ---
USE_CASE_KEYWORDS_LIST: List[str] = []
CONNECTIVITY_PHRASE_MATCHER: Optional[PhraseMatcher] = None
# CATEGORY_PHRASE_MATCHER: Optional[PhraseMatcher] = None # --- REMOVED ---
USE_CASE_PHRASE_MATCHER: Optional[PhraseMatcher] = None
IP_TOKEN_MATCHER: Optional[Matcher] = None

# ==============================================================================
# --- DATA PREPROCESSING & EMBEDDING LOGIC (Phase 0 & 1.5) ---
# ==============================================================================

def clean_text_phase0(text):
    if pd.isna(text) or text == "N/A": return ""
    return str(text).strip()

def parse_connectivity_phase0(conn_str):
    if pd.isna(conn_str) or not conn_str: return []
    conn_str_lower = str(conn_str).lower()
    conn_map = {"lorawan":"LoRaWAN","lora p2p":"LoRa P2P","lora (mesh)":"LoRa Mesh","lora":"LoRa",
                "bluetooth 5.0 (ble)":"BLE","bluetooth (ble)":"BLE","ble":"BLE","bluetooth":"BLE",
                "wi-fi (2.4ghz 802.11b/g/n)":"Wi-Fi","wi-fi (2.4/5ghz 802.11ac)":"Wi-Fi","wi-fi":"Wi-Fi","wifi":"Wi-Fi",
                "ethernet (10/100m)":"Ethernet","ethernet (gbe)":"Ethernet","ethernet":"Ethernet",
                "lte cat-m1":"LTE Cat-M1","lte cat m1":"LTE Cat-M1","cat-m1":"LTE Cat-M1","cat m1":"LTE Cat-M1",
                "lte cat 1":"LTE Cat 1","cat 1":"LTE Cat 1","lte cat 4":"LTE Cat 4","cat 4":"LTE Cat 4",
                "lte (optional":"LTE","lte":"LTE","cellular (optional":"LTE","cellular":"LTE","4g":"LTE","5g":"5G",
                "nb-iot":"NB-IoT","nbiot":"NB-IoT","gps":"GPS","gnss":"GPS",
                "usb type-c":"USB","usb-c":"USB","usb (programming/charging)":"USB","usb (for at commands/fw update)":"USB","usb":"USB",
                "uart":"UART","i2c":"I2C","spi":"SPI","gpio":"GPIO","swd":"SWD","nfc":"NFC","mesh":"Mesh","concentrator adapter":"Concentrator Adapter"}
    extracted = set()
    for k, v in sorted(conn_map.items(), key=lambda item: len(item[0]), reverse=True):
        if k in conn_str_lower: extracted.add(v)
    return sorted(list(extracted))

def run_phase0_preprocessing(excel_file_path: str, output_dir: str) -> pd.DataFrame:
    print(f"--- Running Phase 0: Data Preprocessing for {excel_file_path} (NO CATEGORY COLUMN EXPECTED) ---")
    _PRODUCTS_PKL = os.path.join(output_dir, "df_products_enhanced.pkl")
    _FEATURES_PKL = os.path.join(output_dir, "df_features.pkl")
    _MAPPING_PKL = os.path.join(output_dir, "df_mapping_exploded.pkl")

    if not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created dir: {output_dir}")

    print(f"Loading raw data from {excel_file_path}...")
    try:
        df_products_raw = pd.read_excel(excel_file_path, sheet_name="Product Table")
        df_features_raw = pd.read_excel(excel_file_path, sheet_name="Feature Table")
        df_mapping_raw = pd.read_excel(excel_file_path, sheet_name="Product-Feature Mapping Table") 
        print("Raw data loaded successfully.")
    except FileNotFoundError: print(f"ERROR: Excel file '{excel_file_path}' not found."); raise
    except Exception as e: print(f"ERROR reading Excel file: {e}"); raise
    
    df_products = df_products_raw.copy()
    
    # --- NO Category column standardization needed if we are removing it ---
        
    for col_name in df_products.select_dtypes(include=['object']).columns: 
        df_products[col_name] = df_products[col_name].apply(clean_text_phase0)
    
    df_products.dropna(subset=['Product_ID'], inplace=True)
    df_products['Product_ID'] = df_products['Product_ID'].astype(str).str.strip() 
    df_products['Connectivity_List'] = df_products['Connectivity'].apply(parse_connectivity_phase0) if 'Connectivity' in df_products else [[]]*len(df_products)

    df_features = df_features_raw.copy()
    df_features.dropna(subset=['Feature_ID'], inplace=True); df_features['Feature_ID'] = df_features['Feature_ID'].astype(str).str.strip()
    if 'Description' in df_features.columns and 'Feature_Description' not in df_features.columns:
        df_features.rename(columns={'Description': 'Feature_Description'}, inplace=True)
    for col_name in df_features.select_dtypes(include=['object']).columns: 
        if col_name != 'Feature_ID': df_features[col_name] = df_features[col_name].apply(clean_text_phase0)
    
    df_mapping_cleaned = df_mapping_raw.copy()
    expected_map_pid_col = "Product_ID"; expected_map_fid_col = "Feature_ID"
    # ... (mapping table column name standardization remains the same) ...
    current_map_cols = {}
    for actual_col in df_mapping_cleaned.columns:
        if not expected_map_pid_col in current_map_cols and "product" in actual_col.lower() and "id" in actual_col.lower(): current_map_cols[expected_map_pid_col] = actual_col
        if not expected_map_fid_col in current_map_cols and "feature" in actual_col.lower() and "id" in actual_col.lower(): current_map_cols[expected_map_fid_col] = actual_col
    rename_map_cols_dict = {v_actual: k_expected for k_expected, v_actual in current_map_cols.items() if v_actual != k_expected and v_actual in df_mapping_cleaned.columns}
    if rename_map_cols_dict: df_mapping_cleaned.rename(columns=rename_map_cols_dict, inplace=True)
    if expected_map_pid_col not in df_mapping_cleaned.columns: raise ValueError(f"Mapping Table missing '{expected_map_pid_col}' like column")
    if expected_map_fid_col not in df_mapping_cleaned.columns: raise ValueError(f"Mapping Table missing '{expected_map_fid_col}' like column")
    df_mapping_cleaned.dropna(subset=[expected_map_pid_col, expected_map_fid_col], inplace=True)
    df_mapping_cleaned[expected_map_pid_col] = df_mapping_cleaned[expected_map_pid_col].astype(str).str.strip()
    df_mapping_cleaned[expected_map_fid_col] = df_mapping_cleaned[expected_map_fid_col].astype(str).str.strip()
    df_mapping_exploded = df_mapping_cleaned[[expected_map_pid_col, expected_map_fid_col]].rename(
        columns={expected_map_pid_col: 'Product_ID', expected_map_fid_col: 'Feature_ID_Single'}
    ).drop_duplicates().reset_index(drop=True)
    
    df_features_indexed_temp = df_features.set_index('Feature_ID') 
    feature_desc_col_name = 'Feature_Description' 
    columns_to_merge_from_features = ['Feature_Name']
    if feature_desc_col_name in df_features_indexed_temp.columns:
        columns_to_merge_from_features.append(feature_desc_col_name)
    else: feature_desc_col_name = None 
    prod_feat_details = df_mapping_exploded.merge(df_features_indexed_temp[columns_to_merge_from_features].reset_index(), left_on='Feature_ID_Single', right_on='Feature_ID', how='left')
    prod_to_names = prod_feat_details.groupby('Product_ID')['Feature_Name'].apply(lambda x: sorted(list(x.dropna().unique()))).to_dict()
    df_products['Product_Feature_Names'] = df_products['Product_ID'].map(lambda pid: prod_to_names.get(pid, []))
    if feature_desc_col_name and feature_desc_col_name in prod_feat_details.columns:
        prod_to_descs = prod_feat_details.groupby('Product_ID')[feature_desc_col_name].apply(lambda x: sorted(list(x.dropna().unique()))).to_dict()
        df_products['Product_Feature_Descriptions'] = df_products['Product_ID'].map(lambda pid: prod_to_descs.get(pid, []))
    else: df_products['Product_Feature_Descriptions'] = [[] for _ in range(len(df_products))]

    def get_searchable_text(r): 
        # --- MODIFIED: Removed 'Category' from here ---
        parts = [str(r.get(c,'')) for c in ['Product_Name','Use_Case','Notes']] 
        # If you have a column that *could* contain category-like keywords (e.g., Product_Line), add it here.
        # Example: parts.append(str(r.get('Product_Line', '')))
        parts.append(' '.join(r.get('Product_Feature_Names', []))); parts.append(' '.join(r.get('Product_Feature_Descriptions', [])))
        return " ".join(filter(None, parts)).lower()
    df_products['Searchable_Text'] = df_products.apply(get_searchable_text, axis=1)
    
    # ... (Derived feature functions like extract_power_sources, extract_ip_rating_numeric, etc. remain)
    # Adjust check_custom_firmware and extract_form_factor_keywords if they heavily relied on a specific "Category" string
    def extract_power_sources(text):
        sources=set();tl=text.lower();
        if "battery" in tl:sources.add("battery")
        if "solar" in tl or "solar power compatibility" in tl:sources.add("solar")
        if "poe" in tl or "power over ethernet" in tl or "802.3af" in tl or "802.3at" in tl:sources.add("poe")
        if "usb powered" in tl or ("usb" in tl and ("power" in tl or "powered" in tl)):sources.add("usb_powered")
        if "dc power" in tl or re.search(r'\d+-\d+\s*v\s*dc|\d+\s*v\s*dc|\d+v\s*input',tl):sources.add("dc_power")
        return sorted(list(sources)) if sources else ["unknown"]
    df_products['Derived_Power_Source']=df_products['Searchable_Text'].apply(extract_power_sources)
    def extract_ip_rating_numeric(text):
        tl=text.lower();ratings=re.findall(r'ip(\d{2})',tl);num_r=[int(r) for r in ratings if r.isdigit()]
        if not num_r:
            if "weatherproof" in tl or "outdoor enclosure" in tl:num_r.append(65)
            if "waterproof" in tl or "integrated waterproof" in tl:num_r.append(67)
        return max(num_r) if num_r else None
    df_products['Derived_IP_Rating_Numeric']=df_products['Searchable_Text'].apply(extract_ip_rating_numeric)
    def extract_lorawan_versions(text):
        vers=set();tl=text.lower();vers.update(re.findall(r'lorawan\s*(?:v)?(\d\.\d\.\d(?:\.\d)?)',tl))
        if "lorawan 1.0.3" in tl:vers.add("1.0.3")
        if "lorawan 1.0.2" in tl:vers.add("1.0.2")
        if "lorawan 1.0.4" in tl:vers.add("1.0.4")
        return sorted(list(vers))
    df_products['Derived_LoRaWAN_Versions']=df_products['Searchable_Text'].apply(extract_lorawan_versions)
    
    def check_custom_firmware(text, conn_list, product_name_col_val): # Changed cat_str_val to product_name_col_val
        tl = text.lower(); p_name_lower = str(product_name_col_val).lower()
        kws = ["custom firmware","programmable","sdk","rui3","arduino","platformio","open source"]
        if any(k in tl for k in kws): return True
        # If category is gone, this check becomes less specific.
        # We can check if 'module' appears in product name or searchable text as a heuristic.
        if "module" in p_name_lower and any(c in conn_list for c in["UART","USB","SWD"])and("at command" in tl or "at commands" in tl): return True
        return False
    df_products['Derived_Custom_Firmware']=df_products.apply(lambda r:check_custom_firmware(r['Searchable_Text'],r['Connectivity_List'],r.get('Product_Name','')),axis=1)

    def extract_form_factor_keywords(text, product_name_col_val): # Changed cat_str_val
        kws=set();tl=text.lower(); p_name_lower = str(product_name_col_val).lower()
        if "compact" in tl:kws.add("compact")
        if "small" in tl or "smallest" in tl:kws.add("small")
        if "miniature" in tl:kws.add("miniature")
        if "outdoor enclosure" in tl or "waterproof" in tl or "ip6" in tl: kws.add("enclosure_outdoor_rugged")
        if "sip" in tl:kws.add("sip")
        # Heuristic if category is gone:
        if "breakout" in p_name_lower or "breakout board" in tl :kws.add("breakout_board")
        if "base board" in p_name_lower or "base board" in tl :kws.add("base_board")
        if "module" in p_name_lower and not any(x in kws for x in["sip","breakout_board","base_board"]):kws.add("module_generic")
        return sorted(list(kws))
    df_products['Derived_Form_Factor_Keywords']=df_products.apply(lambda r:extract_form_factor_keywords(r['Searchable_Text'],r.get('Product_Name','')),axis=1)
    
    def create_embedding_text(r):
        # --- MODIFIED: Removed 'Cat' (Category) from here ---
        parts={"Product":r.get('Product_Name',''),"Model":r.get('Product_Model',''), # "Cat":r.get('Category',''),
               "UseCase":r.get('Use_Case',''),"Notes":r.get('Notes',''),"Features":'. '.join(r.get('Product_Feature_Names',[])),
               "Connectivity":','.join(r.get('Connectivity_List',[])),"Power":','.join(r.get('Derived_Power_Source',[])),
               "IP":str(r.get('Derived_IP_Rating_Numeric','N/A')),"LoRaVer":','.join(r.get('Derived_LoRaWAN_Versions',[])),
               "CustomFW":str(r.get('Derived_Custom_Firmware',False)),"FormFactor":','.join(r.get('Derived_Form_Factor_Keywords',[]))}
        return ". ".join(f"{k}: {v}" for k,v in parts.items() if v and str(v).strip() and v!='N/A')
    df_products['Combined_Text_For_Embedding']=df_products.apply(create_embedding_text,axis=1)

    cols_to_keep=['Product_ID','Product_Name','Product_Model', # 'Category' removed from this explicit list
                    'Connectivity_List','Use_Case','Notes','Product_Feature_Names','Product_Feature_Descriptions',
                    'Derived_Power_Source','Derived_IP_Rating_Numeric', 'Searchable_Text',
                    'Derived_LoRaWAN_Versions','Derived_Custom_Firmware','Derived_Form_Factor_Keywords',
                    'Combined_Text_For_Embedding']
    # If 'Category' column still exists in df_products (e.g. wasn't dropped from Excel but you want to ignore it),
    # it will be picked up by opt_orig_cols if we want to keep it for some other reason.
    # For now, we assume it's effectively gone for the model's core logic.
    opt_orig_cols=['Product_Line','Product_Subline','Hardware/Software','Region Support','Price Range']
    if 'Category' in df_products.columns: # If it still exists, add it to optional for saving
        if 'Category' not in opt_orig_cols: opt_orig_cols.append('Category')

    for col in opt_orig_cols:
        if col in df_products.columns and col not in cols_to_keep:cols_to_keep.append(col)
    
    df_products_enhanced=df_products[[c for c in cols_to_keep if c in df_products.columns]].copy()
    
    with open(_PRODUCTS_PKL,"wb")as f:pickle.dump(df_products_enhanced,f)
    with open(_FEATURES_PKL,"wb")as f:pickle.dump(df_features,f) 
    with open(_MAPPING_PKL,"wb")as f:pickle.dump(df_mapping_exploded,f)
    print("Phase 0 preprocessing complete. Files saved.")
    return df_products_enhanced

# ... (run_phase1_5_embeddings remains the same)
def run_phase1_5_embeddings(df_prods_enhanced: pd.DataFrame, output_dir: str, sbert_instance: SentenceTransformer) -> np.ndarray:
    print(f"--- Running Phase 1.5: Generating Product Embeddings ---")
    _EMBEDDINGS_NPY = os.path.join(output_dir, "product_embeddings.npy")
    if 'Combined_Text_For_Embedding' not in df_prods_enhanced.columns: raise KeyError("'Combined_Text_For_Embedding' missing.")
    product_texts = df_prods_enhanced['Combined_Text_For_Embedding'].fillna("").tolist()
    product_embeddings_arr = sbert_instance.encode(product_texts, show_progress_bar=False, convert_to_tensor=False)
    np.save(_EMBEDDINGS_NPY, product_embeddings_arr)
    print(f"Embeddings generated ({len(product_embeddings_arr)}) and saved to {_EMBEDDINGS_NPY}.")
    return product_embeddings_arr

# ==============================================================================
# --- NLP LOGIC (Phase 1) ---
# ==============================================================================
def load_nlp_resources_internal(spacy_model_name=SPACY_MODEL_NAME):
    global nlp_global, CONNECTIVITY_KEYWORDS_DICT, USE_CASE_KEYWORDS_LIST # CATEGORY_KEYWORDS_DICT removed
    global CONNECTIVITY_PHRASE_MATCHER, USE_CASE_PHRASE_MATCHER, IP_TOKEN_MATCHER # CATEGORY_PHRASE_MATCHER removed
    if nlp_global is not None: return
    try: nlp_global = spacy.load(spacy_model_name)
    except OSError: print(f"Downloading spaCy model {spacy_model_name}..."); spacy.cli.download(spacy_model_name); nlp_global = spacy.load(spacy_model_name)
    print(f"spaCy model '{spacy_model_name}' loaded for NLP.")
    
    CONNECTIVITY_KEYWORDS_DICT = { "lorawan":"LoRaWAN","lora":"LoRa","lora p2p":"LoRa P2P","lora mesh":"LoRa Mesh","ble":"BLE","bluetooth":"BLE","bluetooth 5.0":"BLE","wi-fi":"Wi-Fi","wifi":"Wi-Fi","802.11b/g/n":"Wi-Fi","802.11ac":"Wi-Fi","ethernet":"Ethernet","gbe":"Ethernet","lte cat-m1":"LTE Cat-M1","lte cat m1":"LTE Cat-M1","cat-m1":"LTE Cat-M1","cat m1":"LTE Cat-M1","lte cat 1":"LTE Cat 1","cat 1":"LTE Cat 1","lte cat 4":"LTE Cat 4","cat 4":"LTE Cat 4","lte":"LTE","cellular":"LTE","4g":"LTE","5g":"5G","nb-iot":"NB-IoT","nbiot":"NB-IoT","gps":"GPS","gnss":"GPS","usb":"USB","usb-c":"USB","type-c":"USB","uart":"UART","i2c":"I2C","spi":"SPI","gpio":"GPIO","swd":"SWD","nfc":"NFC","mesh":"Mesh" }
    # --- CATEGORY_KEYWORDS_DICT REMOVED ---
    USE_CASE_KEYWORDS_LIST = ["gateway","sensor","module","kit","tracker", # Added common category terms here as general keywords
                              "agriculture","smart farm","farming","smart city","urban","industrial","factory","iiot","asset tracking","tracking","location","environmental monitoring","environment","air quality","water quality","weather station","soil moisture","indoor","building automation","office","smart building","outdoor","field deployment","rural","remote monitoring","remote management","remote control","low power","power efficient","battery powered","battery life","energy efficient","ultra low power","long range","extended range","compact","small size","miniature","small form factor","high performance","robust","reliable","durable","rugged","cost effective","affordable","low cost","customizable","programmable","flexible","open source","security","secure","encryption","easy setup","plug and play","user friendly","quick deployment","pre-configured","qr code setup","data visualization","dashboard","phone app"]
    
    CONNECTIVITY_PHRASE_MATCHER=PhraseMatcher(nlp_global.vocab,attr="LOWER");CONNECTIVITY_PHRASE_MATCHER.add("CONNECTIVITY",[nlp_global.make_doc(k)for k in CONNECTIVITY_KEYWORDS_DICT.keys()])
    # --- CATEGORY_PHRASE_MATCHER REMOVED ---
    USE_CASE_PHRASE_MATCHER=PhraseMatcher(nlp_global.vocab,attr="LOWER");USE_CASE_PHRASE_MATCHER.add("USE_CASE",[nlp_global.make_doc(k)for k in USE_CASE_KEYWORDS_LIST])
    IP_TOKEN_MATCHER=Matcher(nlp_global.vocab);IP_TOKEN_MATCHER.add("IP_TOKEN",[[{"TEXT":{"REGEX":"^IP[0-9]{2}$"}}]])

def extract_requirements_from_text_internal(text: str) -> Dict[str, Any]:
    if nlp_global is None: load_nlp_resources_internal()
    doc = nlp_global(text); original_text_lower = text.lower()
    extracted = {"connectivity_required":set(),
                 "category_specified":set(), # Keep the key, but it won't be populated by specific category matcher
                 "use_case_keywords":set(),
                 "special_features":{"ip_rating_min":None,"ip_rating_exact":None,"power_source":set(),"form_factor":set(),"lorawan_version":None,"custom_firmware":None,"manage_credentials":None},
                 "qualifiers":set(),"original_query":text}
    sf = extracted["special_features"]
    if CONNECTIVITY_PHRASE_MATCHER:
        for _,s,e in CONNECTIVITY_PHRASE_MATCHER(doc): extracted["connectivity_required"].add(CONNECTIVITY_KEYWORDS_DICT[doc[s:e].text.lower()])
    # --- CATEGORY_PHRASE_MATCHER logic REMOVED ---
    if USE_CASE_PHRASE_MATCHER:
        for _,s,e in USE_CASE_PHRASE_MATCHER(doc):
            term = doc[s:e].text.lower(); extracted["use_case_keywords"].add(term)
            # Heuristics based on use_case_keywords that might imply category or form factor
            if term in ["gateway", "gateways", "lorawan gateway"]: extracted["category_specified"].add("Gateway") # Populate from use case
            if term in ["sensor", "sensors", "environmental sensor", "iot sensor"]: extracted["category_specified"].add("Sensor")
            if term in ["module", "modules", "core module", "radio module", "lpwan module"]: extracted["category_specified"].add("Module")
            if term in ["kit", "kits", "starter kit", "development kit", "dev kit"]: extracted["category_specified"].add("Kit")
            if term in ["tracker", "trackers", "asset tracker"]: extracted["category_specified"].add("Tracker")
            if term in ["compact","small size","miniature","small form factor"]:sf["form_factor"].add(term)
            if term in ["low power","power efficient","battery powered","battery life","energy efficient","ultra low power"]:sf["power_source"].add("battery");extracted["qualifiers"].add(term)
    
    # ... (IP Rating, Power, Form Factor, LoRaWAN Version, Firmware, Credentials logic remains the same) ...
    # (Ensure full logic from your refined Cell 2 is here)
    processed_char_indices = set()
    at_least_phrases = r'(?:at\s+least|minimum\s+of|greater\s+than\s+or\s+equal\s+to|at\s+minimum)\s+IP(\d{2})'
    or_higher_phrases = r'IP(\d{2})\s+(?:or\s+higher|or\s+better|minimum|min\b)'
    for match in re.finditer(f"({at_least_phrases})|({or_higher_phrases})", original_text_lower, re.I):
        ip_v_str = match.group(2) or match.group(4);
        if ip_v_str:sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,int(ip_v_str));[processed_char_indices.add(i) for i in range(match.start(),match.end())]
    if IP_TOKEN_MATCHER:
        for _,s,e in IP_TOKEN_MATCHER(doc):
            if doc[s].idx in processed_char_indices: continue
            ip_v_m = re.search(r'(\d{2})',doc[s:e].text)
            if ip_v_m: ip_v=int(ip_v_m.group(1));sf["ip_rating_exact"]=ip_v;sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,ip_v);[processed_char_indices.add(i) for t in doc[s:e] for i in range(t.idx,t.idx+len(t.text))]
    if re.search(r'\bweatherproof\b',original_text_lower):extracted["qualifiers"].add("weatherproof");sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,65)
    if re.search(r'\bwaterproof\b',original_text_lower):extracted["qualifiers"].add("waterproof");sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,67)
    if re.search(r'\bsolar\b(?!\s*panel)|\bsolar\s*power',original_text_lower,re.I):sf["power_source"].add("solar")
    if re.search(r'\bbattery\b|battery life|battery-powered',original_text_lower,re.I):sf["power_source"].add("battery")
    if re.search(r'low-power|low\s*power|power-efficient|energy\s*efficient|ultra\s*low\s*power',original_text_lower,re.I):extracted["qualifiers"].add("low-power");sf["power_source"].add("battery")
    if re.search(r'microamps\s*in\s*sleep|\bµA\s*sleep',original_text_lower,re.I):extracted["qualifiers"].add("ultra-low-power-sleep");sf["power_source"].add("battery")
    ff_m=re.search(r'\b(compact|small|smallest|miniature|small\s*form\s*factor)\b',original_text_lower,re.I)
    if ff_m:t=ff_m.group(1).lower();sf["form_factor"].add(t);extracted["qualifiers"].add(t)
    lora_v_m=re.search(r'LoRaWAN\s*(?:v)?(\d\.\d\.\d(?:\.\d)?)',original_text_lower,re.I)
    if lora_v_m:sf["lorawan_version"]=lora_v_m.group(1)
    if re.search(r'custom firmware|programmable|sdk|\bRUI3\b|develop own firmware',original_text_lower,re.I):sf["custom_firmware"]=True
    # AT commands check (no longer relies on category string from product data)
    if ("module" in original_text_lower or any(cat_kw in extracted["use_case_keywords"] for cat_kw in ["module", "modules"])) and \
       re.search(r'\bAT\s*commands?\b',text,re.I):
        sf["custom_firmware"]=True;extracted["qualifiers"].add("AT commands support")
    if re.search(r'manage\s*(?:my|our|own)\s*(?:LoRaWAN\s+)?credentials\b|manage\s*(?:LoRaWAN\s+)?keys\b',text,re.I):sf["manage_credentials"]=True;sf["custom_firmware"]=True
    
    for k in ["connectivity_required","category_specified","use_case_keywords","qualifiers"]:extracted[k]=sorted(list(extracted[k]))
    for k_sf_ in ["power_source","form_factor"]:sf[k_sf_]=sorted(list(sf[k_sf_])) 
    return extracted

# ==============================================================================
# --- RECOMMENDATION LOGIC (Phase 2) ---
# ==============================================================================
def filter_products_internal(df_prods: pd.DataFrame, extracted_reqs: dict):
    fdf = df_prods.copy(); logs = []
    # --- CATEGORY FILTER REMOVED (or made optional if category_specified is populated by NLP heuristics) ---
    if extracted_reqs["category_specified"]: # This list might be populated by NLP heuristics now
        cat_re = '|'.join([re.escape(c) for c in extracted_reqs["category_specified"]])
        # This assumes df_prods still has a 'Category' column, even if we don't explicitly filter on it as a primary step.
        # If 'Category' column is fully removed from df_prods, this part needs to be removed or adapted.
        # For now, let's assume df_prods MIGHT have a 'Category' column from the original Excel, even if not used for hard filtering.
        # If your df_products_global no longer has 'Category' at all, this will fail.
        # A safer approach if Category column is gone from df_products_global:
        # logs.append(f"Category filter skipped as category module is removed or data unreliable. Specified keywords: {extracted_reqs['category_specified']}")
        # If category_specified by NLP is used for scoring/soft preference, that's handled in score_product.
        # For now, let's assume the 'Category' column MIGHT exist in df_prods for this .str.contains to not error out,
        # but it won't be a primary filter if extracted_reqs["category_specified"] is empty.
        if 'Category' in fdf.columns:
             fdf_before_cat_filter = len(fdf)
             fdf = fdf[fdf['Category'].str.contains(cat_re, case=False, na=False)]
             logs.append(f"F_CatKeywords({extracted_reqs['category_specified']}):{len(fdf)} (from {fdf_before_cat_filter})")
        else:
             logs.append(f"Category column not present in products; skipping NLP category keyword filter. NLP cats: {extracted_reqs['category_specified']}")

    # if fdf.empty and extracted_reqs["category_specified"]: # Only if category was specified and resulted in zero
    #     return pd.DataFrame(), logs

    if extracted_reqs["connectivity_required"]:
        rc = set(c.lower() for c in extracted_reqs["connectivity_required"])
        fdf = fdf[fdf['Connectivity_List'].apply(lambda pcl: isinstance(pcl,list) and rc.issubset(set(c.lower() for c in pcl)))]; logs.append(f"F_Conn({extracted_reqs['connectivity_required']}):{len(fdf)}")
    if fdf.empty: return pd.DataFrame(), logs
    lora_v = extracted_reqs["special_features"].get("lorawan_version")
    if lora_v: fdf = fdf[fdf['Derived_LoRaWAN_Versions'].apply(lambda v: isinstance(v,list) and lora_v in v)]; logs.append(f"F_LoraV({lora_v}):{len(fdf)}")
    if fdf.empty: return pd.DataFrame(), logs
    ip_min = extracted_reqs["special_features"].get("ip_rating_min")
    if ip_min is not None: fdf = fdf[fdf['Derived_IP_Rating_Numeric'].apply(lambda ip: pd.notna(ip) and ip >= ip_min)]; logs.append(f"F_IPMin({ip_min}):{len(fdf)}")
    if fdf.empty: return pd.DataFrame(), logs
    ip_ex = extracted_reqs["special_features"].get("ip_rating_exact")
    if ip_ex is not None: fdf = fdf[fdf['Derived_IP_Rating_Numeric'] == ip_ex]; logs.append(f"F_IPExact({ip_ex}):{len(fdf)}")
    return fdf, logs

def score_product_internal(product_row: pd.Series, product_embedding: Optional[np.ndarray], 
                           extracted_reqs: dict, query_embedding: Optional[np.ndarray], 
                           semantic_threshold: float):
    score = 0; details = []; sf = extracted_reqs["special_features"]; H,S,R,B = SCORE_HARD_REQUIREMENT,SCORE_STRONG_PREFERENCE,SCORE_SOFT_RELEVANCE,SCORE_SEMANTIC_SIMILARITY_BOOST
    
    # --- CATEGORY SCORING MODIFIED ---
    # Score if NLP heuristically identified category keywords and product's 'Category' column (if it exists) matches.
    # Or, score based on NLP category keywords matching product_name/description if 'Category' column is absent.
    product_category_text = str(product_row.get('Category','')).lower() # Get actual category if column exists
    product_searchable_text_for_cat = str(product_row.get('Searchable_Text','')).lower()
    
    if extracted_reqs["category_specified"]:
        matched_nlp_cat = False
        for req_cat_keyword in extracted_reqs["category_specified"]:
            req_cat_keyword_lower = req_cat_keyword.lower()
            if 'Category' in product_row and product_row.get('Category') and req_cat_keyword_lower in product_category_text:
                matched_nlp_cat = True; break
            elif req_cat_keyword_lower in product_searchable_text_for_cat : # Fallback to searching in text
                matched_nlp_cat = True; break
        if matched_nlp_cat:
            score+=R # Score as soft relevance if category is now heuristic
            details.append(f"Implied Cat:{','.join(extracted_reqs['category_specified'])}")

    if extracted_reqs["connectivity_required"]: score+=H*len(extracted_reqs["connectivity_required"]); details.append(f"Conn:{','.join(extracted_reqs['connectivity_required'])}")
    lora_v=sf.get("lorawan_version");
    if lora_v and lora_v in product_row.get('Derived_LoRaWAN_Versions',[]): score+=H; details.append(f"LoRaV {lora_v}")
    ip_m=sf.get("ip_rating_min"); ip_e=sf.get("ip_rating_exact"); p_ip=product_row.get('Derived_IP_Rating_Numeric')
    if pd.notna(p_ip):
        if ip_e is not None and p_ip==ip_e: score+=H; details.append(f"IP Exact:{int(p_ip)}")
        elif ip_m is not None and p_ip>=ip_m: score+=H; details.append(f"IP {int(p_ip)}(Min:{ip_m})")
    if sf.get("custom_firmware") and product_row.get('Derived_Custom_Firmware'):
        if "CustomFW" not in details: score+=H; details.append("CustomFW")
    if sf.get("manage_credentials") and product_row.get('Derived_Custom_Firmware'):
        if not sf.get("custom_firmware") and "ManagesCreds" not in details : score+=H 
        if "ManagesCreds" not in details: details.append("ManagesCreds")
    for ps_m in set(sf.get("power_source",[])).intersection(set(product_row.get('Derived_Power_Source',[]))): score+=S; details.append(f"Pwr:{ps_m.capitalize()}")
    for ff_m in set(sf.get("form_factor",[])).intersection(set(product_row.get('Derived_Form_Factor_Keywords',[]))): score+=S; details.append(f"FF:{ff_m.replace('_',' ').capitalize()}")
    q_match=set(); prod_search_txt=product_row.get('Searchable_Text','').lower()
    for q_k in extracted_reqs.get("qualifiers",[]):
        if q_k.lower() in prod_search_txt and q_k not in q_match: score+=S/2; details.append(f"Qual:{q_k.capitalize()}"); q_match.add(q_k)
    # Use case keywords scoring remains important
    uc_txt=f"{product_row.get('Use_Case','')} {' '.join(product_row.get('Product_Feature_Names',[]))}".lower();uc_c=0
    for uc_kw in extracted_reqs.get("use_case_keywords",[]):
        if uc_kw in uc_txt and uc_c<2: score+=R; details.append(f"UC:{uc_kw.capitalize()}");uc_c+=1
    raw_sim=0.0
    if query_embedding is not None and product_embedding is not None:
        raw_sim = util.cos_sim(query_embedding, product_embedding)[0][0].item()
        if raw_sim >= semantic_threshold: score+=B; details.append(f"SemSim:{raw_sim:.2f}(>{semantic_threshold:.2f})")
    return score, details, raw_sim

def get_recommendations_core_internal(user_query_text: str, df_all_products: pd.DataFrame, 
                                      all_product_embeddings: np.ndarray, sbert_model_for_query: SentenceTransformer, 
                                      top_n=3, current_semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD):
    extracted_reqs = extract_requirements_from_text_internal(user_query_text)
    # df_all_products here is df_products_global which has been standardized (or attempted)
    candidate_products_df, filter_log = filter_products_internal(df_all_products, extracted_reqs)
    
    if candidate_products_df.empty: return [], extracted_reqs, filter_log
    query_embedding = sbert_model_for_query.encode(extracted_reqs['original_query'], convert_to_tensor=False)
    recs_data = []
    for original_idx, product_row in candidate_products_df.iterrows(): 
        product_emb = None
        try:
            # Assuming df_products_global (which is df_all_products) has a consistent index aligning with embeddings
            positional_idx = df_products_global.index.get_loc(original_idx) 
            if positional_idx < len(all_product_embeddings):
                product_emb = all_product_embeddings[positional_idx]
        except KeyError: 
            print(f"Warning: Index {original_idx} for embedding lookup failed.")
            
        final_score, matched_details, raw_similarity = score_product_internal(
            product_row, product_emb, extracted_reqs, query_embedding, current_semantic_threshold
        )
        if final_score > 0:
            reasoning_str = "; ".join(matched_details) or "General match based on filters."
            recs_data.append({
                "product_name": product_row.get('Product_Name'), 
                "product_model": product_row.get('Product_Model', ''),
                "category": product_row.get('Category', 'N/A'), # Get category if col exists, else N/A
                "score": final_score, 
                "reasoning": reasoning_str,
                "similarity_score": float(f"{raw_similarity:.4f}")  })
    ranked = sorted(recs_data, key=lambda x: x["score"], reverse=True)
    return ranked[:top_n], extracted_reqs, filter_log

# ==============================================================================
# --- FastAPI Application ---
# ==============================================================================
app = FastAPI(title="RAKWireless Product Recommender")
origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.1.6:8080", "http://localhost:8080"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserQuery(BaseModel): name: str; email: str; description: str
class RecItem(BaseModel): product_name: str; product_model: Optional[str]=""; category: Optional[str]=""; score: float; reasoning: str; similarity_score: float
class ReqResponse(BaseModel): recommendations: List[RecItem]; submission_id: int; extracted_requirements: Dict[str, Any]
class SelLogItem(BaseModel): name: str; model: str
class SelLog(BaseModel): submission_id: int; name: str; email: str; selected_products: List[SelLogItem]

os.makedirs(LOGS_DIR, exist_ok=True)
SUBMISSION_LOG_FILE = os.path.join(LOGS_DIR, "submissions.jsonl")
SELECTION_LOG_FILE = os.path.join(LOGS_DIR, "selections.jsonl")
def log_to_jsonl(file_path: str, data: dict):
    data["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        with open(file_path, "a") as f: f.write(json.dumps(data) + "\n")
    except Exception as e: print(f"Error logging to {file_path}: {e}")

def needs_reprocessing(source_file_path, target_file_path):
    if not os.path.exists(target_file_path): return True
    if not os.path.exists(source_file_path): print(f"Warning: Source {source_file_path} missing"); return False
    return os.path.getmtime(source_file_path) > os.path.getmtime(target_file_path)

@app.on_event("startup")
async def startup_event_tasks():
    global df_products_global, product_embeddings_global, sbert_model_global, df_features_global, nlp_global
    print("FastAPI App Startup: Initializing resources...")
    os.makedirs(DATA_DIR, exist_ok=True)

    if needs_reprocessing(SOURCE_EXCEL_FILE, PRODUCTS_PKL):
        print(f"Reprocessing Phase 0: {SOURCE_EXCEL_FILE} -> {DATA_DIR}")
        df_products_global = run_phase0_preprocessing(SOURCE_EXCEL_FILE, DATA_DIR)
    else:
        print(f"Loading preprocessed product data from {PRODUCTS_PKL}...")
        with open(PRODUCTS_PKL, "rb") as f: df_products_global = pickle.load(f)
        # --- ENSURE CATEGORY STANDARDIZATION EVEN WHEN LOADING FROM PKL ---
        if df_products_global is not None:
            df_products_global = standardize_category_column(df_products_global) 
            print(f"Ensured 'Category' column (if present) is standardized in df_products_global after PKL load.")
        # --- END ENSURE ---

    if df_products_global is None: raise RuntimeError("CRITICAL: df_products_global is None.")
    print(f"Products loaded/processed: {len(df_products_global)}")

    print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
    sbert_model_global = SentenceTransformer(SBERT_MODEL_NAME)
    print("SBERT model loaded.")

    if needs_reprocessing(PRODUCTS_PKL, EMBEDDINGS_NPY) or not os.path.exists(EMBEDDINGS_NPY):
        print(f"Regenerating Phase 1.5 Embeddings...")
        product_embeddings_global = run_phase1_5_embeddings(df_products_global, DATA_DIR, sbert_model_global)
    elif os.path.exists(EMBEDDINGS_NPY): product_embeddings_global = np.load(EMBEDDINGS_NPY)
    else: raise FileNotFoundError(f"Embeddings {EMBEDDINGS_NPY} not found and regen failed.")
    if product_embeddings_global is None: raise RuntimeError("CRITICAL: product_embeddings_global is None.")
    print(f"Embeddings loaded/generated: {len(product_embeddings_global)}")
    if len(df_products_global) != len(product_embeddings_global): print(f"CRITICAL WARNING: Product/Embedding count mismatch!")
    
    print("Loading NLP (spaCy) resources...")
    load_nlp_resources_internal()
    
    try:
        with open(FEATURES_PKL, "rb") as f: temp_df_features = pickle.load(f)
        if temp_df_features is not None and 'Feature_ID' in temp_df_features.columns:
            df_features_global = temp_df_features.set_index('Feature_ID') 
        print(f"Features loaded: {len(df_features_global) if df_features_global is not None else 'None'}")
    except Exception as e: print(f"Note: Could not load features file {FEATURES_PKL}: {e}")
    print("Application startup sequence complete.")

@app.post("/submit-requirement", response_model=ReqResponse)
async def api_submit_requirement(query: UserQuery, request: Request):
    if not all([df_products_global is not None, product_embeddings_global is not None, 
                sbert_model_global is not None, nlp_global is not None]):
        raise HTTPException(status_code=503, detail="Service not ready.")
    
    recs, ex_reqs, f_log = get_recommendations_core_internal(
        query.description, df_products_global, product_embeddings_global, 
        sbert_model_global, top_n=3, current_semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD)
    sub_id = int(datetime.datetime.utcnow().timestamp())
    log_data = {"submission_id":sub_id,"user_name":query.name,"user_email":query.email,
                "user_description":query.description,"recommendations_generated":[dict(r) for r in recs],
                "extracted_requirements_log":ex_reqs,"filter_log":" ".join(f_log),
                "user_ip":request.client.host if request.client else "N/A","user_agent":request.headers.get("user-agent","N/A")}
    log_to_jsonl(SUBMISSION_LOG_FILE,log_data)
    return ReqResponse(recommendations=recs,submission_id=sub_id,extracted_requirements=ex_reqs)

@app.post("/log-selection")
async def api_log_selection(sel_data: SelLog, request: Request):
    log_data = {"submission_id":sel_data.submission_id,"user_name":sel_data.name,"user_email":sel_data.email,
                "selected_products":[item.model_dump() for item in sel_data.selected_products],
                "user_ip":request.client.host if request.client else "N/A","user_agent":request.headers.get("user-agent","N/A")}
    log_to_jsonl(SELECTION_LOG_FILE,log_data)
    return {"status":"selection logged","redirect_url":STORE_URL}