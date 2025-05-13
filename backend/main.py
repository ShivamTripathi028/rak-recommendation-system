# rak_recommender_local/main.py

# --- Standard Library Imports ---
from dotenv import load_dotenv
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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # <<< --- NEW IMPORT FOR CORS ---

try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
except ImportError:
    print("spaCy not found. Please ensure it's in requirements.txt and installed.")
    raise
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("sentence-transformers not found. Please ensure it's in requirements.txt and installed.")
    raise

# --- Application Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") 
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

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

df_products_global: Optional[pd.DataFrame] = None
df_features_global: Optional[pd.DataFrame] = None
product_embeddings_global: Optional[np.ndarray] = None
nlp_global: Optional[spacy.language.Language] = None
sbert_model_global: Optional[SentenceTransformer] = None

CONNECTIVITY_KEYWORDS_DICT: Dict[str, str] = {}
CATEGORY_KEYWORDS_DICT: Dict[str, str] = {}
USE_CASE_KEYWORDS_LIST: List[str] = []
CONNECTIVITY_PHRASE_MATCHER: Optional[PhraseMatcher] = None
CATEGORY_PHRASE_MATCHER: Optional[PhraseMatcher] = None
USE_CASE_PHRASE_MATCHER: Optional[PhraseMatcher] = None
IP_TOKEN_MATCHER: Optional[Matcher] = None

# ==============================================================================
# --- DATA PREPROCESSING & EMBEDDING LOGIC (Phase 0 & 1.5) ---
# (Your full functions for run_phase0_preprocessing and run_phase1_5_embeddings go here)
# ==============================================================================
def clean_text_phase0(text):
    if pd.isna(text) or text == "N/A": return ""
    return str(text).strip()

def parse_connectivity_phase0(conn_str):
    if pd.isna(conn_str) or not conn_str: return []
    conn_str_lower = str(conn_str).lower()
    conn_map = {
        "lorawan": "LoRaWAN", "lora p2p": "LoRa P2P", "lora (mesh)": "LoRa Mesh", "lora": "LoRa",
        "bluetooth 5.0 (ble)": "BLE", "bluetooth (ble)": "BLE", "ble": "BLE", "bluetooth": "BLE",
        "wi-fi (2.4ghz 802.11b/g/n)": "Wi-Fi", "wi-fi (2.4/5ghz 802.11ac)": "Wi-Fi", "wi-fi": "Wi-Fi", "wifi": "Wi-Fi",
        "ethernet (10/100m)": "Ethernet", "ethernet (gbe)": "Ethernet", "ethernet": "Ethernet",
        "lte cat-m1": "LTE Cat-M1", "lte cat m1": "LTE Cat-M1", "cat-m1": "LTE Cat-M1", "cat m1": "LTE Cat-M1",
        "lte cat 1": "LTE Cat 1", "cat 1": "LTE Cat 1", "lte cat 4": "LTE Cat 4", "cat 4": "LTE Cat 4",
        "lte (optional": "LTE", "lte": "LTE", "cellular (optional": "LTE", "cellular": "LTE",
        "nb-iot": "NB-IoT", "nbiot": "NB-IoT", "gps": "GPS", "usb type-c": "USB", "usb-c": "USB",
        "usb (programming/charging)": "USB", "usb (for at commands/fw update)": "USB", "usb": "USB",
        "uart": "UART", "i2c": "I2C", "spi": "SPI", "gpio": "GPIO", "swd": "SWD", "nfc": "NFC",
        "mesh": "Mesh", "concentrator adapter": "Concentrator Adapter"
    }
    extracted = set()
    for k, v in sorted(conn_map.items(), key=lambda item: len(item[0]), reverse=True):
        if k in conn_str_lower: extracted.add(v)
    return sorted(list(extracted))

def parse_feature_ids_phase0(ids_str):
    if pd.isna(ids_str) or not str(ids_str).strip(): return []
    return [fid.strip() for fid in str(ids_str).split(',') if fid.strip()]

def run_phase0_preprocessing(excel_file_path: str, output_dir: str) -> pd.DataFrame:
    print(f"--- Running Phase 0: Data Preprocessing for {excel_file_path} ---")
    _PRODUCTS_PKL = os.path.join(output_dir, "df_products_enhanced.pkl")
    _FEATURES_PKL = os.path.join(output_dir, "df_features.pkl")
    _MAPPING_PKL = os.path.join(output_dir, "df_mapping_exploded.pkl")

    if not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created dir: {output_dir}")

    print(f"Loading raw data from {excel_file_path}...")
    df_products_raw = pd.read_excel(excel_file_path, sheet_name="Product Table")
    df_features_raw = pd.read_excel(excel_file_path, sheet_name="Feature Table")
    df_mapping_raw = pd.read_excel(excel_file_path, sheet_name="Product-Feature Mapping Table")
    
    df_products = df_products_raw.copy()
    for col in df_products.select_dtypes(include=['object']).columns: df_products[col] = df_products[col].apply(clean_text_phase0)
    df_products.dropna(subset=['Product_ID'], inplace=True); df_products['Product_ID'] = df_products['Product_ID'].astype(int)
    df_products['Connectivity_List'] = df_products['Connectivity'].apply(parse_connectivity_phase0) if 'Connectivity' in df_products else [[]]*len(df_products)

    df_features = df_features_raw.copy()
    df_features.dropna(subset=['Feature_ID'], inplace=True); df_features['Feature_ID'] = df_features['Feature_ID'].astype(str).str.strip()
    for col in df_features.select_dtypes(include=['object']).columns: 
        if col != 'Feature_ID': df_features[col] = df_features[col].apply(clean_text_phase0)
    
    df_mapping = df_mapping_raw.copy()
    if 'Feature ID' in df_mapping: df_mapping.rename(columns={'Feature ID': 'Feature_IDs_Str'}, inplace=True)
    elif 'Feature_ID' in df_mapping: df_mapping.rename(columns={'Feature_ID': 'Feature_IDs_Str'}, inplace=True)
    else: raise ValueError("Feature ID column missing in Mapping Table")
    if 'Product ID' in df_mapping: df_mapping.rename(columns={'Product ID': 'Product_ID'}, inplace=True)
    elif 'Product_ID' not in df_mapping: raise ValueError("Product ID column missing in Mapping Table")
    df_mapping.dropna(subset=['Product_ID', 'Feature_IDs_Str'], inplace=True); df_mapping['Product_ID'] = df_mapping['Product_ID'].astype(int)
    df_mapping_exploded = df_mapping.assign(FIDs=df_mapping['Feature_IDs_Str'].apply(parse_feature_ids_phase0)).explode('FIDs').rename(columns={'FIDs':'Feature_ID_Single'})
    df_mapping_exploded = df_mapping_exploded[df_mapping_exploded['Feature_ID_Single'] != ''][['Product_ID', 'Feature_ID_Single']].drop_duplicates().reset_index(drop=True)

    df_features_indexed_temp = df_features.set_index('Feature_ID')
    prod_feat_details = df_mapping_exploded.merge(df_features_indexed_temp[['Feature_Name', 'Description']].reset_index(), left_on='Feature_ID_Single', right_on='Feature_ID', how='left')
    prod_to_names = prod_feat_details.groupby('Product_ID')['Feature_Name'].apply(lambda x: sorted(list(x.dropna().unique()))).to_dict()
    prod_to_descs = prod_feat_details.groupby('Product_ID')['Description'].apply(lambda x: sorted(list(x.dropna().unique()))).to_dict()
    df_products['Product_Feature_Names'] = df_products['Product_ID'].map(lambda pid: prod_to_names.get(pid, []))
    df_products['Product_Feature_Descriptions'] = df_products['Product_ID'].map(lambda pid: prod_to_descs.get(pid, []))

    def get_searchable_text(r):
        parts = [str(r.get(c,'')) for c in ['Product_Name','Category','Use_Case','Notes']]
        parts.append(' '.join(r.get('Product_Feature_Names', []))); parts.append(' '.join(r.get('Product_Feature_Descriptions', [])))
        return " ".join(filter(None, parts)).lower()
    df_products['Searchable_Text'] = df_products.apply(get_searchable_text, axis=1)

    def extract_power_sources(text):
        sources = set(); text_lower = text.lower()
        if "battery" in text_lower: sources.add("battery")
        if "solar" in text_lower or "solar power compatibility" in text_lower: sources.add("solar")
        if "poe" in text_lower or "power over ethernet" in text_lower or "802.3af" in text_lower or "802.3at" in text_lower: sources.add("poe")
        if "usb powered" in text_lower or ("usb" in text_lower and ("power" in text_lower or "powered" in text_lower)): sources.add("usb_powered")
        if "dc power" in text_lower or re.search(r'\d+-\d+\s*v\s*dc|\d+\s*v\s*dc|\d+v\s*input', text_lower): sources.add("dc_power")
        return sorted(list(sources)) if sources else ["unknown"]
    df_products['Derived_Power_Source'] = df_products['Searchable_Text'].apply(extract_power_sources)
    def extract_ip_rating_numeric(text):
        ratings = re.findall(r'ip(\d{2})', text.lower()); num_ratings = [int(r) for r in ratings if r.isdigit()]
        if not num_ratings:
            if "weatherproof" in text.lower() or "outdoor enclosure" in text.lower() or "industrial-grade enclosure" in text.lower(): num_ratings.append(65)
            if "waterproof" in text.lower() or "integrated waterproof enclosure" in text.lower(): num_ratings.append(67)
        return max(num_ratings) if num_ratings else None
    df_products['Derived_IP_Rating_Numeric'] = df_products['Searchable_Text'].apply(extract_ip_rating_numeric)
    def extract_lorawan_versions(text):
        versions = set(); text_lower = text.lower()
        versions.update(re.findall(r'lorawan\s*(?:v)?(\d\.\d\.\d(?:\.\d)?)', text_lower))
        if "lorawan 1.0.3" in text_lower: versions.add("1.0.3")
        if "lorawan 1.0.2" in text_lower: versions.add("1.0.2")
        if "lorawan 1.0.4" in text_lower: versions.add("1.0.4")
        return sorted(list(set(versions)))
    df_products['Derived_LoRaWAN_Versions'] = df_products['Searchable_Text'].apply(extract_lorawan_versions)
    def check_custom_firmware(text, conn_list, cat_str):
        text_lower = text.lower(); cat_lower = cat_str.lower()
        keywords = ["custom firmware", "programmable mcu", "sdk", "rui3", "arduino", "platformio", "customizable firmware", "open source firmware"]
        if any(k in text_lower for k in keywords): return True
        if "module" in cat_lower and any(c in conn_list for c in ["UART", "USB", "SWD"]) and ("at command" in text_lower or "at commands" in text_lower): return True
        return False
    df_products['Derived_Custom_Firmware'] = df_products.apply(lambda r: check_custom_firmware(r['Searchable_Text'], r['Connectivity_List'], str(r.get('Category', ''))), axis=1)
    def extract_form_factor_keywords(text, cat_str):
        kws = set(); text_lower = text.lower(); cat_lower = cat_str.lower()
        if "compact" in text_lower: kws.add("compact")
        if "small" in text_lower or "smallest" in text_lower: kws.add("small")
        if "miniature" in text_lower: kws.add("miniature")
        if "outdoor enclosure" in text_lower or "industrial-grade enclosure" in text_lower or "waterproof enclosure" in text_lower or "ip6" in text_lower: kws.add("enclosure_outdoor_rugged")
        if "sip" in text_lower or "system-in-package" in text_lower: kws.add("sip")
        if "breakout board" in cat_lower: kws.add("breakout_board")
        if "base board" in cat_lower: kws.add("base_board")
        if "module" in cat_lower and not any(x in kws for x in ["sip", "breakout_board", "base_board"]): kws.add("module_generic")
        return sorted(list(kws))
    df_products['Derived_Form_Factor_Keywords'] = df_products.apply(lambda r: extract_form_factor_keywords(r['Searchable_Text'], str(r.get('Category', ''))), axis=1)
    def create_embedding_text(r):
        parts = {"Product": r.get('Product_Name',''), "Model": r.get('Product_Model',''), "Category": r.get('Category',''),
                 "Use Case": r.get('Use_Case',''), "Notes": r.get('Notes',''), "Features": '. '.join(r.get('Product_Feature_Names',[])),
                 "Connectivity": ', '.join(r.get('Connectivity_List',[])), "Power": ', '.join(r.get('Derived_Power_Source',[])),
                 "IP": str(r.get('Derived_IP_Rating_Numeric','N/A')), "LoRaVer": ', '.join(r.get('Derived_LoRaWAN_Versions',[])),
                 "CustomFW": str(r.get('Derived_Custom_Firmware',False)), "FormFactor": ', '.join(r.get('Derived_Form_Factor_Keywords',[]))}
        return ". ".join(f"{k}: {v}" for k,v in parts.items() if v and v!='N/A' and str(v).strip())
    df_products['Combined_Text_For_Embedding'] = df_products.apply(create_embedding_text, axis=1)
    df_products_enhanced = df_products[['Product_ID', 'Product_Name', 'Product_Model', 'Category', 'Connectivity_List', 'Use_Case', 'Notes',
                                        'Product_Feature_Names', 'Derived_Power_Source', 'Derived_IP_Rating_Numeric', 'Searchable_Text',
                                        'Derived_LoRaWAN_Versions', 'Derived_Custom_Firmware', 'Derived_Form_Factor_Keywords',
                                        'Combined_Text_For_Embedding']].copy()
    with open(_PRODUCTS_PKL, "wb") as f: pickle.dump(df_products_enhanced, f)
    with open(_FEATURES_PKL, "wb") as f: pickle.dump(df_features, f)
    with open(_MAPPING_PKL, "wb") as f: pickle.dump(df_mapping_exploded, f)
    print("Phase 0 preprocessing complete. Files saved.")
    return df_products_enhanced

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
    global nlp_global, CONNECTIVITY_KEYWORDS_DICT, CATEGORY_KEYWORDS_DICT, USE_CASE_KEYWORDS_LIST
    global CONNECTIVITY_PHRASE_MATCHER, CATEGORY_PHRASE_MATCHER, USE_CASE_PHRASE_MATCHER, IP_TOKEN_MATCHER
    if nlp_global is not None: return
    try: nlp_global = spacy.load(spacy_model_name)
    except OSError: print(f"Downloading spaCy model {spacy_model_name}..."); spacy.cli.download(spacy_model_name); nlp_global = spacy.load(spacy_model_name)
    print(f"spaCy model '{spacy_model_name}' loaded for NLP.")
    CONNECTIVITY_KEYWORDS_DICT = { "lorawan": "LoRaWAN", "lora": "LoRa", "ble": "BLE", "bluetooth": "BLE", "wi-fi": "Wi-Fi", "wifi": "Wi-Fi", "ethernet": "Ethernet", "lte cat-m1": "LTE Cat-M1", "lte cat m1": "LTE Cat-M1", "cat-m1": "LTE Cat-M1", "cat m1": "LTE Cat-M1", "lte cat 1": "LTE Cat 1", "cat 1": "LTE Cat 1", "lte cat 4": "LTE Cat 4", "cat 4": "LTE Cat 4", "lte": "LTE", "cellular": "LTE", "nb-iot": "NB-IoT", "nbiot": "NB-IoT", "gps": "GPS", "usb": "USB", "uart": "UART", "i2c": "I2C", "spi": "SPI", "gpio": "GPIO", "swd": "SWD", "nfc": "NFC", "mesh": "Mesh" }
    CATEGORY_KEYWORDS_DICT = { "gateway": "Gateway", "gateways": "Gateway", "sensor": "Sensor", "sensors": "Sensor", "environmental sensor": "Sensor", "module": "Module", "modules": "Module", "core module": "Module", "radio module": "Module", "kit": "Kit", "kits": "Kit", "starter kit": "Kit", "development kit": "Kit", "tracker": "Tracker", "trackers": "Tracker", "breakout board": "Development Board", "breakout": "Development Board", "evaluation board": "Development Board", "dev board": "Development Board", "base board": "Base Board", "node": "Node", "repeater": "Repeater" }
    USE_CASE_KEYWORDS_LIST = [ "agriculture", "smart farm", "farming", "smart city", "urban", "industrial", "factory", "asset tracking", "tracking", "location", "environmental monitoring", "environment", "air quality", "water quality", "weather", "indoor", "building", "office", "outdoor", "field", "rural", "remote monitoring", "remote management", "low power", "power efficient", "battery powered", "battery life", "long range", "compact", "small size", "miniature", "high performance", "robust", "reliable", "durable", "cost effective", "affordable", "customizable", "programmable", "security", "secure" ]
    CONNECTIVITY_PHRASE_MATCHER = PhraseMatcher(nlp_global.vocab, attr="LOWER"); CONNECTIVITY_PHRASE_MATCHER.add("CONNECTIVITY", [nlp_global.make_doc(k) for k in CONNECTIVITY_KEYWORDS_DICT.keys()])
    CATEGORY_PHRASE_MATCHER = PhraseMatcher(nlp_global.vocab, attr="LOWER"); CATEGORY_PHRASE_MATCHER.add("CATEGORY", [nlp_global.make_doc(k) for k in CATEGORY_KEYWORDS_DICT.keys()])
    USE_CASE_PHRASE_MATCHER = PhraseMatcher(nlp_global.vocab, attr="LOWER"); USE_CASE_PHRASE_MATCHER.add("USE_CASE", [nlp_global.make_doc(k) for k in USE_CASE_KEYWORDS_LIST])
    IP_TOKEN_MATCHER = Matcher(nlp_global.vocab); IP_TOKEN_MATCHER.add("IP_TOKEN", [[{"TEXT": {"REGEX": "^IP[0-9]{2}$"}}]])

def extract_requirements_from_text_internal(text: str) -> Dict[str, Any]:
    if nlp_global is None: load_nlp_resources_internal()
    doc = nlp_global(text); original_text_lower = text.lower()
    extracted = {"connectivity_required":set(),"category_specified":set(),"use_case_keywords":set(),
                 "special_features":{"ip_rating_min":None,"ip_rating_exact":None,"power_source":set(),"form_factor":set(),"lorawan_version":None,"custom_firmware":None,"manage_credentials":None},
                 "qualifiers":set(),"original_query":text}
    sf = extracted["special_features"]
    for _,s,e in CONNECTIVITY_PHRASE_MATCHER(doc): extracted["connectivity_required"].add(CONNECTIVITY_KEYWORDS_DICT[doc[s:e].text.lower()])
    for _,s,e in CATEGORY_PHRASE_MATCHER(doc): extracted["category_specified"].add(CATEGORY_KEYWORDS_DICT[doc[s:e].text.lower()])
    for _,s,e in USE_CASE_PHRASE_MATCHER(doc):
        term = doc[s:e].text.lower(); extracted["use_case_keywords"].add(term)
        if term in ["compact","small size","miniature"]:sf["form_factor"].add(term)
        if term in ["low power","power efficient","battery powered","battery life"]:sf["power_source"].add("battery");extracted["qualifiers"].add(term)
    processed_char_indices = set()
    at_least_phrases = r'(?:at\s+least|minimum\s+of|greater\s+than\s+or\s+equal\s+to|at\s+minimum)\s+IP(\d{2})'
    or_higher_phrases = r'IP(\d{2})\s+(?:or\s+higher|or\s+better|minimum|min\b)'
    for match in re.finditer(f"({at_least_phrases})|({or_higher_phrases})", original_text_lower, re.I):
        ip_v_str = match.group(2) or match.group(4)
        if ip_v_str:sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,int(ip_v_str));[processed_char_indices.add(i) for i in range(match.start(),match.end())]
    for _,s,e in IP_TOKEN_MATCHER(doc):
        if doc[s].idx in processed_char_indices: continue
        ip_v_m = re.search(r'(\d{2})',doc[s:e].text)
        if ip_v_m: ip_v=int(ip_v_m.group(1));sf["ip_rating_exact"]=ip_v;sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,ip_v);[processed_char_indices.add(i) for t in doc[s:e] for i in range(t.idx,t.idx+len(t.text))]
    if re.search(r'\bweatherproof\b',original_text_lower):extracted["qualifiers"].add("weatherproof");sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,65)
    if re.search(r'\bwaterproof\b',original_text_lower):extracted["qualifiers"].add("waterproof");sf["ip_rating_min"]=max(sf["ip_rating_min"] or 0,67)
    if re.search(r'\bsolar\b(?!\s*panel)|\bsolar\s*power',original_text_lower,re.I):sf["power_source"].add("solar")
    if re.search(r'\bbattery\b|battery life|battery-powered',original_text_lower,re.I):sf["power_source"].add("battery")
    if re.search(r'low-power|low\s*power|power-efficient',original_text_lower,re.I):extracted["qualifiers"].add("low-power");sf["power_source"].add("battery")
    if re.search(r'microamps\s*in\s*sleep|\bµA\s*sleep',original_text_lower,re.I):extracted["qualifiers"].add("ultra-low-power-sleep");sf["power_source"].add("battery")
    ff_m=re.search(r'\b(compact|small|smallest|miniature)\b',original_text_lower,re.I)
    if ff_m:t=ff_m.group(1).lower();sf["form_factor"].add(t);extracted["qualifiers"].add(t)
    lora_v_m=re.search(r'LoRaWAN\s*(?:v)?(\d\.\d\.\d(?:\.\d)?)',original_text_lower,re.I)
    if lora_v_m:sf["lorawan_version"]=lora_v_m.group(1)
    if re.search(r'custom firmware|programmable|sdk|\bRUI3\b|develop own firmware',original_text_lower,re.I):sf["custom_firmware"]=True
    cat_list=list(extracted["category_specified"]);cat_str_lower="".join(cat_list).lower()
    if "module" in cat_str_lower and re.search(r'\bAT\s*commands?\b',text,re.I):sf["custom_firmware"]=True;extracted["qualifiers"].add("AT commands support")
    if re.search(r'manage\s*(?:my|our|own)\s*(?:LoRaWAN\s+)?credentials\b|manage\s*(?:LoRaWAN\s+)?keys\b',text,re.I):sf["manage_credentials"]=True;sf["custom_firmware"]=True
    for k in ["connectivity_required","category_specified","use_case_keywords","qualifiers"]:extracted[k]=sorted(list(extracted[k]))
    for k_sf in ["power_source","form_factor"]:sf[k_sf]=sorted(list(sf[k_sf]))
    return extracted

# ==============================================================================
# --- RECOMMENDATION LOGIC (Phase 2) ---
# ==============================================================================
def filter_products_internal(df_prods: pd.DataFrame, extracted_reqs: dict):
    fdf = df_prods.copy(); logs = []
    if extracted_reqs["category_specified"]:
        cat_re = '|'.join([re.escape(c) for c in extracted_reqs["category_specified"]])
        fdf = fdf[fdf['Category'].str.contains(cat_re, case=False, na=False)]; logs.append(f"F_Cat({extracted_reqs['category_specified']}):{len(fdf)}")
    if fdf.empty: return pd.DataFrame(), logs
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
    if extracted_reqs["category_specified"] and any(rc.lower() in str(product_row.get('Category','')).lower() for rc in extracted_reqs["category_specified"]): score+=H; details.append(f"Cat:{','.join(extracted_reqs['category_specified'])}")
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
    candidate_products_df, filter_log = filter_products_internal(df_all_products, extracted_reqs)
    if candidate_products_df.empty: return [], extracted_reqs, filter_log
    query_embedding = sbert_model_for_query.encode(extracted_reqs['original_query'], convert_to_tensor=False)
    recs_data = []
    # Ensure 'original_idx' is preserved if candidate_products_df is a slice
    # If candidate_products_df.index are the original indices from df_all_products, this is fine.
    for original_idx, product_row in candidate_products_df.iterrows():
        # Map original_idx from df_all_products to its positional equivalent in all_product_embeddings
        try:
            # If df_all_products was reset_index() before embedding, use positional index.
            # If it maintained its original index, use get_loc.
            # Assuming df_all_products index aligns with all_product_embeddings sequence
            positional_idx = df_all_products.index.get_loc(original_idx) 
            product_emb = all_product_embeddings[positional_idx] if positional_idx < len(all_product_embeddings) else None
        except KeyError: # If original_idx from candidate_df isn't in df_all_products (should not happen if filtered correctly)
            print(f"Warning: Index {original_idx} not found in df_products_global.index for embedding lookup.")
            product_emb = None
            
        final_score, matched_details, raw_similarity = score_product_internal(
            product_row, product_emb, extracted_reqs, query_embedding, current_semantic_threshold
        )
        if final_score > 0:
            reasoning_str = "; ".join(matched_details) or "General match based on filters."
            recs_data.append({
                "product_name": product_row.get('Product_Name'), "product_model": product_row.get('Product_Model', ''),
                "category": product_row.get('Category'), "score": final_score, "reasoning": reasoning_str,
                "similarity_score": float(f"{raw_similarity:.4f}")  })
    ranked = sorted(recs_data, key=lambda x: x["score"], reverse=True)
    return ranked[:top_n], extracted_reqs, filter_log

# ==============================================================================
# --- FastAPI Application ---
# ==============================================================================
app = FastAPI(title="RAKWireless Product Recommender")

# <<< --- ADD CORS MIDDLEWARE --- >>>
origins = [
    "http://localhost:5173",      # Default Vite dev server for React
    "http://127.0.0.1:5173",    # Also common for Vite
    "http://localhost:3000",      # Common create-react-app dev port
    "http://127.0.0.1:3000",    # Also common for CRA
    "http://192.168.1.6:8080",  # <<< --- ADD THIS LINE --- >>>
    # Add other origins if needed, e.g., your deployed frontend URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# <<< --- END CORS MIDDLEWARE --- >>>


# app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# --- Pydantic Models ---
class UserQuery(BaseModel): name: str; email: str; description: str
class RecItem(BaseModel): product_name: str; product_model: Optional[str]=""; category: Optional[str]=""; score: float; reasoning: str; similarity_score: float
class ReqResponse(BaseModel): recommendations: List[RecItem]; submission_id: int; extracted_requirements: Dict[str, Any]
class SelLogItem(BaseModel): name: str; model: str
class SelLog(BaseModel): submission_id: int; name: str; email: str; selected_products: List[SelLogItem]

# --- Logging ---
os.makedirs(LOGS_DIR, exist_ok=True)
SUBMISSION_LOG_FILE = os.path.join(LOGS_DIR, "submissions.jsonl")
SELECTION_LOG_FILE = os.path.join(LOGS_DIR, "selections.jsonl")
def log_to_jsonl(file_path: str, data: dict):
    data["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        with open(file_path, "a") as f: f.write(json.dumps(data) + "\n")
    except Exception as e: print(f"Error logging to {file_path}: {e}")

# --- Startup Event ---
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
        with open(PRODUCTS_PKL, "rb") as f: df_products_global = pickle.load(f)
    print(f"Products: {len(df_products_global) if df_products_global is not None else 'None'}")

    print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
    sbert_model_global = SentenceTransformer(SBERT_MODEL_NAME)
    print("SBERT model loaded.")

    if df_products_global is not None and (needs_reprocessing(PRODUCTS_PKL, EMBEDDINGS_NPY) or not os.path.exists(EMBEDDINGS_NPY)):
        print(f"Regenerating Phase 1.5 Embeddings: {PRODUCTS_PKL} -> {EMBEDDINGS_NPY}")
        product_embeddings_global = run_phase1_5_embeddings(df_products_global, DATA_DIR, sbert_model_global)
    elif os.path.exists(EMBEDDINGS_NPY):
        product_embeddings_global = np.load(EMBEDDINGS_NPY)
    else:
        print(f"CRITICAL ERROR: Embeddings file {EMBEDDINGS_NPY} missing and could not be regenerated.")
        raise FileNotFoundError(f"Embeddings {EMBEDDINGS_NPY} not found and regeneration pre-conditions not met.")
    print(f"Embeddings: {len(product_embeddings_global) if product_embeddings_global is not None else 'None'}")

    if df_products_global is not None and product_embeddings_global is not None and len(df_products_global) != len(product_embeddings_global):
        print(f"CRITICAL WARNING: Product ({len(df_products_global)}) / Embedding ({len(product_embeddings_global)}) count mismatch!")
    
    print("Loading NLP (spaCy) resources...")
    load_nlp_resources_internal()
    
    try:
        with open(FEATURES_PKL, "rb") as f: df_features_global = pickle.load(f)
        if df_features_global is not None and 'Feature_ID' in df_features_global.columns and df_features_global.index.name != 'Feature_ID':
            df_features_global.set_index('Feature_ID', inplace=True)
        print(f"Features: {len(df_features_global) if df_features_global is not None else 'None'}")
    except Exception as e: print(f"Note: Could not load features file {FEATURES_PKL}: {e}")
    print("Application startup sequence complete.")

# --- API Endpoints ---
# @app.get("/", response_class=HTMLResponse)
# async def root():
#     return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/submit-requirement", response_model=ReqResponse)
async def api_submit_requirement(query: UserQuery, request: Request):
    if not all([df_products_global is not None, product_embeddings_global is not None, 
                sbert_model_global is not None, nlp_global is not None]):
        raise HTTPException(status_code=503, detail="Service not ready. Models/data loading.")
    
    recs, ex_reqs, f_log = get_recommendations_core_internal(
        query.description, df_products_global, product_embeddings_global, 
        sbert_model_global, top_n=3, current_semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD
    )
    sub_id = int(datetime.datetime.utcnow().timestamp())
    log_data = {
        "submission_id": sub_id, "user_name": query.name, "user_email": query.email,
        "user_description": query.description, "recommendations_generated": [dict(r) for r in recs], # Ensure serializable
        "extracted_requirements_log": ex_reqs, "filter_log": " ".join(f_log),
        "user_ip": request.client.host if request.client else "N/A", 
        "user_agent": request.headers.get("user-agent", "N/A")
    }
    log_to_jsonl(SUBMISSION_LOG_FILE, log_data)
    return ReqResponse(recommendations=recs, submission_id=sub_id, extracted_requirements=ex_reqs)

@app.post("/log-selection")
async def api_log_selection(sel_data: SelLog, request: Request):
    # Convert SelLogItem Pydantic models to dicts for logging
    selected_products_list = [item.model_dump() for item in sel_data.selected_products]
    log_data = {
        "submission_id": sel_data.submission_id, "user_name": sel_data.name, "user_email": sel_data.email,
        "selected_products": selected_products_list,
        "user_ip": request.client.host if request.client else "N/A", 
        "user_agent": request.headers.get("user-agent", "N/A")
    }
    log_to_jsonl(SELECTION_LOG_FILE, log_data)
    return {"status": "selection logged", "redirect_url": STORE_URL}

# To run: uvicorn main:app --reload
# Ensure RAKWireless Recommender Dataset.xlsx is in the same directory as main.py
# Ensure static/index.html, static/script.js exist