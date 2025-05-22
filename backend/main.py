# backend/main.py

# --- Standard Library Imports ---
import os
import pickle
import re
import json
import datetime
from typing import List, Dict, Any, Optional, Set

# --- Third-Party Imports ---
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

SCORE_HARD_REQUIREMENT_MET = 150
SCORE_EXPLICIT_FEATURE_MATCH = 120
SCORE_STRONG_PREFERENCE_MET = 70
SCORE_SOFT_RELEVANCE_MET = 35
SCORE_SEMANTIC_SIMILARITY_BOOST = 80
DEFAULT_SEMANTIC_THRESHOLD = 0.55

# --- Global Variables ---
df_products_global: Optional[pd.DataFrame] = None
df_features_global: Optional[pd.DataFrame] = None
product_embeddings_global: Optional[np.ndarray] = None
nlp_global: Optional[spacy.language.Language] = None
sbert_model_global: Optional[SentenceTransformer] = None

CONNECTIVITY_KEYWORDS_NLP_DICT: Dict[str, str] = {}
CATEGORY_KEYWORDS_NLP_DICT: Dict[str, str] = {}
USE_CASE_KEYWORDS_NLP_LIST: List[str] = []
CONNECTIVITY_PHRASE_MATCHER_NLP: Optional[PhraseMatcher] = None
CATEGORY_PHRASE_MATCHER_NLP: Optional[PhraseMatcher] = None
USE_CASE_PHRASE_MATCHER_NLP: Optional[PhraseMatcher] = None
IP_TOKEN_MATCHER_NLP: Optional[Matcher] = None

JSON_CONNECTIVITY_MAP_GLOBAL: Dict[str, str] = {}
JSON_POWER_MAP_GLOBAL: Dict[str, str] = {}


# ==============================================================================
# --- DATA PREPROCESSING & EMBEDDING LOGIC (Phase 0 & 1.5) ---
# ==============================================================================
def clean_text_phase0(text):
    if pd.isna(text) or text == "N/A": return ""
    return str(text).strip()

def parse_connectivity_phase0(conn_str: str) -> List[str]:
    if pd.isna(conn_str) or not conn_str: return []
    conn_str_lower = str(conn_str).lower()
    # This map should be comprehensive for terms found in your EXCEL's "Connectivity" column
    conn_map = {
        "lorawan": "LoRaWAN", "lora p2p": "LoRa P2P", "lora (mesh)": "LoRa Mesh", "lora": "LoRa",
        "bluetooth 5.0 (ble)": "BLE", "bluetooth (ble)": "BLE", "ble": "BLE", "bluetooth": "BLE", # Prioritize BLE if "bluetooth" mentioned
        "wi-fi (2.4ghz 802.11b/g/n)": "Wi-Fi", "wi-fi (2.4/5ghz 802.11ac)": "Wi-Fi", "wi-fi": "Wi-Fi", "wifi": "Wi-Fi",
        "ethernet (10/100m)": "Ethernet", "ethernet (gbe)": "Ethernet", "ethernet": "Ethernet",
        "lte cat-m1": "LTE Cat-M1", "lte cat m1": "LTE Cat-M1", "cat-m1": "LTE Cat-M1", "cat m1": "LTE Cat-M1", "lte-m": "LTE-M",
        "lte cat 1": "LTE Cat 1", "cat 1": "LTE Cat 1", "lte cat 4": "LTE Cat 4", "cat 4": "LTE Cat 4",
        "lte (optional": "LTE", "lte": "LTE", "cellular (optional": "LTE", "cellular": "LTE", "4g": "LTE", "5g": "5G",
        "nb-iot": "NB-IoT", "nbiot": "NB-IoT",
        "gps": "GPS", "gnss": "GPS", # Often used interchangeably for basic GPS
        "usb type-c": "USB", "usb-c": "USB", "usb (programming/charging)": "USB",
        "usb (for at commands/fw update)": "USB", "usb": "USB",
        "uart": "UART", "i2c": "I2C", "spi": "SPI", "gpio": "GPIO", "swd": "SWD", "nfc": "NFC", "uwb": "UWB",
        "mesh": "Mesh", "concentrator adapter": "Concentrator Adapter", "poe": "PoE",
        "rs485": "RS485", "modbus": "RS485", # Assuming Modbus often implies RS485 in context
        "sdi-12": "SDI-12", "can": "CAN", "lin": "LIN", "mqtt": "MQTT",
        "adc": "Analog Input", # Or "ADC" if you want to be specific
        "analog input": "Analog Input", "digital i/o": "GPIO", # Mapping to more common term
        "aggregation gateway": "AGW", # If AGW is your standard term
        "lpwan": "LPWAN"
    }
    extracted = set()
    # First, check for direct matches of standardized terms (values in conn_map)
    for std_term_val in sorted(list(set(conn_map.values())), key=len, reverse=True):
        if re.search(r'\b' + re.escape(std_term_val.lower()) + r'\b', conn_str_lower):
            extracted.add(std_term_val)

    # Then, iterate through the map keys (raw terms)
    for k_raw, v_std in sorted(conn_map.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(r'\b' + re.escape(k_raw) + r'\b', conn_str_lower): # Use word boundaries for keys too
            extracted.add(v_std)
    return sorted(list(extracted))


def parse_region_support_phase0(region_str: str) -> Dict[str, List[str]]:
    if pd.isna(region_str) or not region_str:
        return {"regions": [], "frequency_bands": []}
    regions_set = set()
    frequency_bands_set = set()
    parts = [p.strip() for p in str(region_str).split(',')]
    for part in parts:
        if not part: continue
        freq_match = re.match(r'^([A-Z]{2}\d{3}(-\d{1,2})?([A-Z])?)$', part.upper()) # Adjusted to catch things like AS923-1, US915A
        if freq_match:
            frequency_bands_set.add(freq_match.group(1))
        elif part.upper() == "GLOBAL":
            regions_set.add("Global")
        elif len(part) > 3 or re.match(r'^[A-Za-z\s]+(?:/[A-Za-z\s]+)*$', part): # Heuristic for region names, allow slashes
            regions_set.add(part.title().replace(' And ', ' and '))
    text_lower = region_str.lower()
    common_bands_map = {"in865": "IN865", "eu868": "EU868", "us915": "US915", "au915": "AU915", "as923": "AS923", "kr920": "KR920", "ru864": "RU864"}
    for k_band, v_band in common_bands_map.items():
        if k_band in text_lower:
            frequency_bands_set.add(v_band)
    return {"regions": sorted(list(regions_set)), "frequency_bands": sorted(list(frequency_bands_set))}

def extract_power_sources_phase0(text_and_connectivity_list: tuple) -> List[str]:
    text, conn_list_val = text_and_connectivity_list
    sources = set()
    text_lower = str(text).lower()
    # Ensure conn_list_val is a list of strings
    conn_list_lower = [str(c).lower() for c in conn_list_val if isinstance(c, str)] if isinstance(conn_list_val, list) else []


    if "battery" in text_lower: sources.add("battery")
    if "solar" in text_lower or "solar power compatibility" in text_lower: sources.add("solar")
    if "poe" in text_lower or "power over ethernet" in text_lower or "802.3af" in text_lower or "802.3at" in text_lower: sources.add("poe")
    if "poe" in conn_list_lower: sources.add("poe")
    if "usb powered" in text_lower or ("usb" in text_lower and ("power" in text_lower or "powered" in text_lower)): sources.add("usb_powered")
    if "dc power" in text_lower or re.search(r'\d+-\d+\s*v\s*dc|\d+\s*v\s*dc|\d+v\s*input|dc input', text_lower): sources.add("dc_power")
    if "ac power" in text_lower or "mains power" in text_lower or re.search(r'\d+-\d+\s*v\s*ac', text_lower): sources.add("ac_power")

    return sorted(list(sources)) if sources else ["unknown"]

def extract_ip_rating_numeric_phase0(text_and_deployment_env: tuple) -> Optional[int]:
    text, dep_env = text_and_deployment_env
    text_lower = str(text).lower()
    dep_env_lower = str(dep_env).lower() if pd.notna(dep_env) else ""
    ratings = re.findall(r'ip(\d{2})', text_lower)
    num_ratings = [int(r) for r in ratings if r.isdigit()]

    if not num_ratings:
        if "outdoor" in dep_env_lower:
            if "weatherproof" in text_lower or "outdoor enclosure" in text_lower: num_ratings.append(65)
            elif "waterproof" in text_lower or "integrated waterproof" in text_lower: num_ratings.append(67)
            else: num_ratings.append(54)
        elif "industrial" in dep_env_lower and "indoor" not in dep_env_lower :
             num_ratings.append(54)
        if not num_ratings:
            if "weatherproof" in text_lower or "outdoor enclosure" in text_lower: num_ratings.append(65)
            if "waterproof" in text_lower or "integrated waterproof" in text_lower: num_ratings.append(67)
    return max(num_ratings) if num_ratings else None

def extract_lorawan_versions_phase0(text: str) -> List[str]:
    versions = set()
    text_lower = str(text).lower()
    versions.update(re.findall(r'lorawan\s*(?:v)?(\d\.\d\.\d(?:\.\d)?)', text_lower))
    if "lorawan 1.0.3" in text_lower: versions.add("1.0.3")
    if "lorawan 1.0.2" in text_lower: versions.add("1.0.2")
    if "lorawan 1.0.4" in text_lower: versions.add("1.0.4")
    if not versions and "lorawan" in text_lower: versions.add("Generic LoRaWAN") # Default if "lorawan" mentioned but no version
    return sorted(list(set(versions)))

def check_custom_firmware_phase0(text: str, conn_list: List[str], product_name_val: str, category_val: str) -> bool:
    text_lower = str(text).lower()
    prod_name_lower = str(product_name_val).lower() if pd.notna(product_name_val) else ""
    cat_lower = str(category_val).lower() if pd.notna(category_val) else ""
    
    keywords = ["custom firmware", "programmable mcu", "sdk", "rui3", "arduino", "platformio", "customizable firmware", "open source firmware", "api access", "openwrt"]
    if any(k in text_lower for k in keywords): return True
    
    is_module_like = "module" in prod_name_lower or "module" in cat_lower or "breakout" in prod_name_lower # Include breakout boards
    has_dev_interfaces = any(c in conn_list for c in ["UART", "USB", "SWD", "SPI", "I2C"]) # Broader dev interfaces
    supports_at_commands = "at command" in text_lower or "at commands" in text_lower

    if is_module_like and has_dev_interfaces and supports_at_commands: return True
    if "development kit" in cat_lower or "dev kit" in prod_name_lower : return True # Dev kits usually allow custom firmware
    return False

def extract_form_factor_keywords_phase0(text: str, product_name_val: str, category_val: str, dep_env_val: str) -> List[str]:
    kws = set()
    text_lower = str(text).lower()
    prod_name_lower = str(product_name_val).lower() if pd.notna(product_name_val) else ""
    cat_lower = str(category_val).lower() if pd.notna(category_val) else ""
    dep_env_lower = str(dep_env_val).lower() if pd.notna(dep_env_val) else ""


    if "compact" in text_lower: kws.add("compact")
    if "small" in text_lower or "smallest" in text_lower: kws.add("small")
    if "miniature" in text_lower: kws.add("miniature")
    if "outdoor enclosure" in text_lower or "industrial-grade enclosure" in text_lower or \
       "waterproof enclosure" in text_lower or "ip6" in text_lower or "outdoor" in dep_env_lower:
        kws.add("enclosure_outdoor_rugged")
    if "sip" in text_lower or "system-in-package" in text_lower: kws.add("sip")
    if "din rail" in text_lower: kws.add("din_rail_mountable")
    if "raspberry pi hat" in text_lower or "pi hat" in text_lower: kws.add("pi_hat")


    if "breakout board" in cat_lower or "breakout" in prod_name_lower: kws.add("breakout_board")
    if "base board" in cat_lower or "base board" in prod_name_lower: kws.add("base_board")
    if ("module" in cat_lower or "module" in prod_name_lower) and not any(x in kws for x in ["sip", "breakout_board", "base_board", "pi_hat"]):
        kws.add("module_generic")
    return sorted(list(kws))

def run_phase0_preprocessing_json_input(excel_file_path: str, output_dir: str) -> pd.DataFrame:
    print(f"--- Running Phase 0 (JSON Input variant): Data Preprocessing for {excel_file_path} ---")
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
    expected_product_cols_map = {
        'Product_ID': ['Product_ID', 'Product ID'], 'Product_Name': ['Product_Name', 'Product Name'],
        'Product_Model': ['Product_Model', 'Product Model'], 'Product_Line': ['Product_Line', 'Product Line'],
        'Hardware_Software': ['Hardware/Software', 'Hardware Software', 'Hardware_Software'],
        'Deployment_Environment': ['Deployment Environment', 'Deployment_Environment'],
        'Category': ['Category'],
        'Use_Case_Description': ['Use Case & Description', 'Use_Case_Description', 'Description', 'Use_Case'],
        'Connectivity': ['Connectivity'], 'Region_Support': ['Region Support', 'Region_Support'],
        'Notes': ['Notes']
    }
    for target_col, source_options in expected_product_cols_map.items():
        found_col = None
        for source_col in source_options:
            if source_col in df_products.columns: found_col = source_col; break
        if found_col and found_col != target_col: df_products.rename(columns={found_col: target_col}, inplace=True)
        elif not found_col and target_col not in df_products.columns:
            df_products[target_col] = "" # Create empty if missing and generally expected
            print(f"Warning: Column for '{target_col}' not found in Product Table. Created as empty.")

    for col_name in df_products.select_dtypes(include=['object']).columns:
        if col_name in df_products.columns:
            df_products[col_name] = df_products[col_name].apply(clean_text_phase0)

    df_products.dropna(subset=['Product_ID'], inplace=True)
    df_products['Product_ID'] = df_products['Product_ID'].astype(str).str.strip()
    df_products['Connectivity_List'] = df_products['Connectivity'].apply(parse_connectivity_phase0)

    parsed_regions = df_products['Region_Support'].apply(parse_region_support_phase0)
    df_products['Supported_Regions'] = parsed_regions.apply(lambda x: x['regions'])
    df_products['Supported_Frequency_Bands'] = parsed_regions.apply(lambda x: x['frequency_bands'])

    df_features = df_features_raw.copy()
    if 'Feature ID' in df_features.columns and 'Feature_ID' not in df_features.columns:
        df_features.rename(columns={'Feature ID': 'Feature_ID'}, inplace=True)
    df_features.dropna(subset=['Feature_ID'], inplace=True); df_features['Feature_ID'] = df_features['Feature_ID'].astype(str).str.strip()
    if 'Feature_Description' not in df_features.columns and 'Description' in df_features.columns:
        df_features.rename(columns={'Description': 'Feature_Description'}, inplace=True)
    if 'Feature_Description' not in df_features.columns:
        df_features['Feature_Description'] = ""; print("Warning: 'Feature_Description' created as empty in features.")
    for col_name in df_features.select_dtypes(include=['object']).columns:
        if col_name != 'Feature_ID' and col_name in df_features.columns:
            df_features[col_name] = df_features[col_name].apply(clean_text_phase0)

    df_mapping = df_mapping_raw.copy()
    if 'Product ID' in df_mapping.columns: df_mapping.rename(columns={'Product ID': 'Product_ID'}, inplace=True)
    if 'Feature ID' in df_mapping.columns: df_mapping.rename(columns={'Feature ID': 'Feature_ID'}, inplace=True)
    if 'Product_ID' not in df_mapping.columns or 'Feature_ID' not in df_mapping.columns:
        raise ValueError("Product_ID or Feature_ID missing in Mapping Table.")
    df_mapping.dropna(subset=['Product_ID', 'Feature_ID'], inplace=True)
    df_mapping['Product_ID'] = df_mapping['Product_ID'].astype(str).str.strip()
    df_mapping['Feature_ID'] = df_mapping['Feature_ID'].astype(str).str.strip()
    df_mapping_exploded = df_mapping[['Product_ID', 'Feature_ID']].rename(
        columns={'Feature_ID': 'Feature_ID_Single'}
    ).drop_duplicates().reset_index(drop=True)

    df_features_indexed_temp = df_features.set_index('Feature_ID')
    prod_feat_details = df_mapping_exploded.merge(
        df_features_indexed_temp[['Feature_Name', 'Feature_Description']].reset_index(),
        left_on='Feature_ID_Single', right_on='Feature_ID', how='left'
    )
    prod_to_names = prod_feat_details.groupby('Product_ID')['Feature_Name'].apply(lambda x: sorted(list(x.dropna().unique()))).to_dict()
    prod_to_descs = prod_feat_details.groupby('Product_ID')['Feature_Description'].apply(lambda x: sorted(list(x.dropna().unique()))).to_dict()
    df_products['Product_Feature_Names'] = df_products['Product_ID'].map(lambda pid: prod_to_names.get(pid, []))
    df_products['Product_Feature_Descriptions'] = df_products['Product_ID'].map(lambda pid: prod_to_descs.get(pid, []))

    def get_searchable_text_phase0(row):
        text_parts = [
            str(row.get('Product_Name', '')), str(row.get('Product_Model', '')),
            str(row.get('Category', '')), str(row.get('Use_Case_Description', '')),
            str(row.get('Deployment_Environment', '')), str(row.get('Notes', '')),
            ' '.join(row.get('Product_Feature_Names', [])), ' '.join(row.get('Product_Feature_Descriptions', []))
        ]
        return " ".join(filter(None, text_parts)).lower()
    df_products['Searchable_Text'] = df_products.apply(get_searchable_text_phase0, axis=1)

    df_products['Derived_Power_Source'] = df_products.apply(
        lambda row: extract_power_sources_phase0((row['Searchable_Text'], row['Connectivity_List'])), axis=1)
    df_products['Derived_IP_Rating_Numeric'] = df_products.apply(
        lambda row: extract_ip_rating_numeric_phase0((row['Searchable_Text'], row.get('Deployment_Environment', ''))), axis=1)
    df_products['Derived_LoRaWAN_Versions'] = df_products['Searchable_Text'].apply(extract_lorawan_versions_phase0)
    df_products['Derived_Custom_Firmware'] = df_products.apply(
        lambda r: check_custom_firmware_phase0(r['Searchable_Text'], r['Connectivity_List'], r.get('Product_Name',''), r.get('Category','')), axis=1)
    df_products['Derived_Form_Factor_Keywords'] = df_products.apply(
        lambda r: extract_form_factor_keywords_phase0(r['Searchable_Text'], r.get('Product_Name',''), r.get('Category',''), r.get('Deployment_Environment','')), axis=1)

    df_products['Combined_Text_For_Embedding'] = df_products.apply(
        lambda r: (
            f"Product: {r.get('Product_Name', '')}. Model: {r.get('Product_Model', '')}. "
            f"Category: {r.get('Category', '')}. Line: {r.get('Product_Line', '')}. "
            f"Deployment: {r.get('Deployment_Environment', '')}. "
            f"Description: {r.get('Use_Case_Description', '')}. Notes: {r.get('Notes', '')}. "
            f"Features: {'. '.join(r.get('Product_Feature_Names', []))}. "
            f"Connectivity: {', '.join(r.get('Connectivity_List', []))}. "
            f"Power: {', '.join(r.get('Derived_Power_Source',[]))}. "
            f"IP Rating: {r.get('Derived_IP_Rating_Numeric', 'N/A')}. "
            f"LoRaWAN Versions: {', '.join(r.get('Derived_LoRaWAN_Versions',[]))}. "
            f"Custom Firmware: {r.get('Derived_Custom_Firmware', False)}. "
            f"Form Factor: {', '.join(r.get('Derived_Form_Factor_Keywords',[]))}. "
            f"Regions: {', '.join(r.get('Supported_Regions',[]))}. "
            f"Frequency Bands: {', '.join(r.get('Supported_Frequency_Bands',[]))}."
        ), axis=1
    )
    
    cols_to_keep = [
        'Product_ID', 'Product_Name', 'Product_Model', 'Product_Line',
        'Hardware_Software', 'Deployment_Environment', 'Category', 'Use_Case_Description',
        'Connectivity', 'Connectivity_List', 'Region_Support', 'Supported_Regions',
        'Supported_Frequency_Bands', 'Notes', 'Product_Feature_Names', 'Product_Feature_Descriptions',
        'Derived_Power_Source', 'Derived_IP_Rating_Numeric', 'Derived_LoRaWAN_Versions',
        'Derived_Custom_Firmware', 'Derived_Form_Factor_Keywords',
        'Combined_Text_For_Embedding', 'Searchable_Text'
    ]
    optional_original_cols = ['Product_Subline', 'Price Range', 'Product Link'] # Added 'Product Link'
    for col in optional_original_cols:
        if col in df_products.columns and col not in cols_to_keep:
            cols_to_keep.append(col)

    df_products_enhanced = df_products[[c for c in cols_to_keep if c in df_products.columns]].copy()

    with open(_PRODUCTS_PKL, "wb") as f: pickle.dump(df_products_enhanced, f)
    with open(_FEATURES_PKL, "wb") as f: pickle.dump(df_features.reset_index(), f)
    with open(_MAPPING_PKL, "wb")as f:pickle.dump(df_mapping_exploded,f)
    print(f"Phase 0 (JSON variant) preprocessing complete. {len(df_products_enhanced)} products processed. Files saved to {output_dir}.")
    return df_products_enhanced

def run_phase1_5_embeddings(df_prods_enhanced: pd.DataFrame, output_dir: str, sbert_instance: SentenceTransformer) -> np.ndarray:
    print(f"--- Running Phase 1.5: Generating Product Embeddings ---")
    _EMBEDDINGS_NPY = os.path.join(output_dir, "product_embeddings.npy")
    if 'Combined_Text_For_Embedding' not in df_prods_enhanced.columns:
        raise KeyError("'Combined_Text_For_Embedding' missing from product data. Check Phase 0.")
    product_texts = df_prods_enhanced['Combined_Text_For_Embedding'].fillna("").tolist()
    product_embeddings_arr = sbert_instance.encode(product_texts, show_progress_bar=True, convert_to_tensor=False)
    np.save(_EMBEDDINGS_NPY, product_embeddings_arr)
    print(f"Embeddings generated ({len(product_embeddings_arr)}) and saved to {_EMBEDDINGS_NPY}.")
    return product_embeddings_arr

# ==============================================================================
# --- NLP & JSON REQUIREMENT EXTRACTION LOGIC (Phase 1 for JSON) ---
# ==============================================================================
def load_nlp_resources_internal(spacy_model_name=SPACY_MODEL_NAME):
    global nlp_global, CONNECTIVITY_KEYWORDS_NLP_DICT, CATEGORY_KEYWORDS_NLP_DICT, USE_CASE_KEYWORDS_NLP_LIST
    global CONNECTIVITY_PHRASE_MATCHER_NLP, CATEGORY_PHRASE_MATCHER_NLP, USE_CASE_PHRASE_MATCHER_NLP, IP_TOKEN_MATCHER_NLP
    global JSON_CONNECTIVITY_MAP_GLOBAL, JSON_POWER_MAP_GLOBAL

    if nlp_global is not None: return
    try: nlp_global = spacy.load(spacy_model_name)
    except OSError:
        print(f"Downloading spaCy model {spacy_model_name}...");
        spacy.cli.download(spacy_model_name) # type: ignore
        nlp_global = spacy.load(spacy_model_name)
    print(f"spaCy model '{spacy_model_name}' loaded for NLP on 'additionalDetails'.")

    CONNECTIVITY_KEYWORDS_NLP_DICT = {
        "lorawan": "LoRaWAN", "lora": "LoRa", "ble": "BLE", "bluetooth": "BLE",
        "wi-fi": "Wi-Fi", "wifi": "Wi-Fi", "ethernet": "Ethernet", "gps": "GPS",
        "lte": "LTE", "cellular": "LTE", "nb-iot": "NB-IoT", "usb": "USB", "uart":"UART",
        "i2c": "I2C", "spi": "SPI", "rs485": "RS485", "poe": "PoE", "nfc": "NFC", "uwb": "UWB",
        "sdi-12": "SDI-12", "can": "CAN", "lin": "LIN", "mqtt": "MQTT"
    }
    CATEGORY_KEYWORDS_NLP_DICT = {
        "gateway": "Gateway", "sensor": "Sensor", "module": "Module", "kit": "Kit", "tracker": "Tracker",
        "dev kit": "Development Kit", "breakout": "Breakout Board", "base board": "Base Board", "hat": "Pi Hat"
    }
    USE_CASE_KEYWORDS_NLP_LIST = [
        "agriculture", "smart city", "industrial", "asset tracking", "environmental monitoring",
        "air quality", "low power", "long range", "compact", "customizable", "weatherproof",
        "ip65", "ip67", "indoor positioning", "smart building", "logistics", "supply chain"
    ]

    if nlp_global:
        CONNECTIVITY_PHRASE_MATCHER_NLP=PhraseMatcher(nlp_global.vocab,attr="LOWER")
        CONNECTIVITY_PHRASE_MATCHER_NLP.add("CONNECTIVITY_NLP",[nlp_global.make_doc(k)for k in CONNECTIVITY_KEYWORDS_NLP_DICT.keys()])

        CATEGORY_PHRASE_MATCHER_NLP=PhraseMatcher(nlp_global.vocab,attr="LOWER")
        CATEGORY_PHRASE_MATCHER_NLP.add("CATEGORY_NLP",[nlp_global.make_doc(k)for k in CATEGORY_KEYWORDS_NLP_DICT.keys()])

        USE_CASE_PHRASE_MATCHER_NLP=PhraseMatcher(nlp_global.vocab,attr="LOWER")
        USE_CASE_PHRASE_MATCHER_NLP.add("USE_CASE_NLP",[nlp_global.make_doc(k)for k in USE_CASE_KEYWORDS_NLP_LIST])

        IP_TOKEN_MATCHER_NLP=Matcher(nlp_global.vocab)
        IP_TOKEN_MATCHER_NLP.add("IP_TOKEN_NLP",[[{"TEXT":{"REGEX":"^IP[0-9]{2}$"}}]])

    # --- THIS IS THE CRUCIAL MAP TO COMPLETE ---
    JSON_CONNECTIVITY_MAP_GLOBAL = {
        # Wireless Communication
        "lorawan_protocol": "LoRaWAN", "lora_p2p": "LoRa P2P", "lora_basic": "LoRa",
        "meshtastic": "LoRa Mesh", # Or "Mesh" if that's your standard
        "wifi": "Wi-Fi",
        "bluetooth_classic": "BLE", # Standardizing to BLE as per your parse_connectivity_phase0
        "ble": "BLE",
        "nfc": "NFC", "uwb": "UWB",
        "lte_4g_5g": "LTE", # Covers 4G/5G. If 5G is distinct, map 5g_nr separately.
        "lte_m": "LTE-M",
        "nb_iot": "NB-IoT",
        "cat_m1": "LTE Cat-M1", # Your parse_connectivity_phase0 also maps "lte cat-m1" to "LTE Cat-M1"
        "gsm_2g": "GSM", # Or "2G" if that's the standard term
        "5g_nr": "5G", # If you treat 5G as distinct from general LTE
        "agw": "AGW", # Or map to "Aggregation Gateway" if that's the standard
        "lpwan_generic": "LPWAN",
        # GNSS / GPS
        "gps_basic": "GPS",
        "gnss_advanced": "GPS", # Or "GNSS" if your product data uses this.
        # Wired Interfaces
        "ethernet_wired": "Ethernet",
        "poe": "PoE",
        "usb_generic": "USB",
        "usb_specific": "USB", # Assuming different USB types still map to a general "USB" feature
        "pcie": "PCIe",
        "twisted_pair": "Twisted Pair", # This is a cable type, map to a feature if relevant or omit
        "coaxial_cable": "Coaxial Cable", # Same as above
        # Protocols / Data Buses
        "i2c_bus": "I2C",
        "spi_bus": "SPI",
        "uart_serial": "UART",
        "rs485_serial": "RS485",
        "sdi12_protocol": "SDI-12",
        "can_bus": "CAN",
        "lin_bus": "LIN",
        "mqtt_protocol": "MQTT",
        # Sensors / IO
        "adc_io": "Analog Input", # Or "ADC"
        "digital_io": "GPIO",
        "analog_io": "Analog Input",
        "gpio_pins": "GPIO",
        # Add any other IDs from your elaborateConnectivityOptionsData in formUtils.ts
    }
    JSON_POWER_MAP_GLOBAL = {
        "DC Power": "dc_power",
        "AC Power": "ac_power", # This should map to a term your extract_power_sources_phase0 derives
        "Battery Powered": "battery",
        "USB Powered": "usb_powered",
        "Solar Power": "solar",
        "PoE (Power over Ethernet)": "poe",
        "Other / Not Specified": "unknown" # Or a specific flag
    }

def nlp_extract_from_free_text_internal(text: str, existing_requirements: Dict[str, Any]) -> Dict[str, Any]:
    if not text or nlp_global is None: return existing_requirements
    doc = nlp_global(text); original_text_lower = text.lower()
    sf = existing_requirements["special_features"]

    if CONNECTIVITY_PHRASE_MATCHER_NLP:
        for _,s,e in CONNECTIVITY_PHRASE_MATCHER_NLP(doc):
            term = doc[s:e].text.lower()
            std_term = CONNECTIVITY_KEYWORDS_NLP_DICT.get(term)
            if std_term and std_term not in existing_requirements["connectivity_required"]:
                existing_requirements["connectivity_required"].add(std_term)

    if CATEGORY_PHRASE_MATCHER_NLP:
        for _,s,e in CATEGORY_PHRASE_MATCHER_NLP(doc):
            term = doc[s:e].text.lower()
            std_term = CATEGORY_KEYWORDS_NLP_DICT.get(term)
            # Check against the set before adding
            if std_term and isinstance(existing_requirements.get("category_specified"), set) and std_term not in existing_requirements["category_specified"]:
                 existing_requirements["category_specified"].add(std_term)

    if USE_CASE_PHRASE_MATCHER_NLP:
        for _,s,e in USE_CASE_PHRASE_MATCHER_NLP(doc):
            term = doc[s:e].text.lower(); existing_requirements["use_case_keywords"].add(term)
            if term in ["compact","small","miniature"] and isinstance(sf.get("form_factor"), set): sf["form_factor"].add(term)
            if term in ["low power","battery powered"] and isinstance(sf.get("power_source"), set) : sf["power_source"].add("battery"); existing_requirements["qualifiers"].add(term)


    processed_ip_indices = set()
    ip_min_regex = r'(?:at\s+least|minimum\s+of|min\b|>=)\s*IP(\d{2})|IP(\d{2})\s*(?:or\s+higher|or\s+better|\+)'
    for match in re.finditer(ip_min_regex, original_text_lower, re.I):
        ip_val_str = match.group(1) or match.group(2)
        if ip_val_str:
            sf["ip_rating_min"] = max(sf.get("ip_rating_min") or 0, int(ip_val_str))
            for i in range(match.start(), match.end()): processed_ip_indices.add(i)
    if IP_TOKEN_MATCHER_NLP:
        for _, start_tok, end_tok in IP_TOKEN_MATCHER_NLP(doc):
            if doc[start_tok].idx in processed_ip_indices: continue
            ip_val_m = re.search(r'(\d{2})', doc[start_tok:end_tok].text)
            if ip_val_m:
                ip_val = int(ip_val_m.group(1))
                sf["ip_rating_min"] = max(sf.get("ip_rating_min") or 0, ip_val)
    if re.search(r'\bweatherproof\b',original_text_lower): existing_requirements["qualifiers"].add("weatherproof"); sf["ip_rating_min"]=max(sf.get("ip_rating_min") or 0,65)
    if re.search(r'\bwaterproof\b',original_text_lower): existing_requirements["qualifiers"].add("waterproof"); sf["ip_rating_min"]=max(sf.get("ip_rating_min") or 0,67)

    lora_v_m=re.search(r'LoRaWAN\s*(?:v)?(\d\.\d\.\d(?:\.\d)?)',original_text_lower,re.I)
    if lora_v_m and not sf.get("lorawan_version"): sf["lorawan_version"]=lora_v_m.group(1)

    custom_fw_keywords = r'custom firmware|programmable|sdk|\bRUI3\b|develop.*firmware|openwrt'
    manage_creds_keywords = r'manage\s*(?:my|our|own)\s*(?:LoRaWAN\s+)?(?:credentials|keys)\b'
    if re.search(custom_fw_keywords, original_text_lower, re.I):
        if sf.get("custom_firmware") is None: sf["custom_firmware"]=True
    if re.search(manage_creds_keywords, text, re.I):
        sf["manage_credentials"]=True
        if sf.get("custom_firmware") is None: sf["custom_firmware"]=True
    return existing_requirements


def extract_requirements_from_json_internal(json_input_data: Dict[str, Any]) -> Dict[str, Any]:
    reqs: Dict[str, Any] = {
        "region_selected": None, "frequency_band_required": None, "deployment_environment": None,
        "application_type": None, "application_subtypes_features": set(),
        "connectivity_required": set(), "power_source_required": set(),
        "lorawan_type_preference": None, "use_case_keywords": set(),
        "category_specified": set(), # Initialized as a set
        "special_features": {
            "ip_rating_min": None, "ip_rating_exact": None, "form_factor": set(),
            "lorawan_version": None, "custom_firmware": None, "manage_credentials": None,
            "power_source": set() # For NLP derived power needs
        },
        "qualifiers": set(), "additional_details_text": "",
        "original_json_query": json_input_data # Store original query for logging/debugging
    }

    if "region" in json_input_data and isinstance(json_input_data["region"], dict):
        reqs["region_selected"] = json_input_data["region"].get("selected")
        reqs["frequency_band_required"] = json_input_data["region"].get("frequencyBand")

    if "deployment" in json_input_data and isinstance(json_input_data["deployment"], dict):
        reqs["deployment_environment"] = json_input_data["deployment"].get("environment")

    if "application" in json_input_data and isinstance(json_input_data["application"], dict):
        app_info = json_input_data["application"]
        reqs["application_type"] = app_info.get("type")
        if reqs["application_type"]: reqs["use_case_keywords"].add(str(reqs["application_type"]).lower())

        subtypes = app_info.get("subtypes", [])
        if isinstance(subtypes, list):
            for subtype in subtypes:
                if subtype: reqs["application_subtypes_features"].add(str(subtype)); reqs["use_case_keywords"].add(str(subtype).lower())
        other_subtype = app_info.get("otherSubtype", "")
        if other_subtype:
            reqs["application_subtypes_features"].add(str(other_subtype))
            reqs["use_case_keywords"].add(str(other_subtype).lower())
            json_input_data["additionalDetails"] = f"{other_subtype}. {json_input_data.get('additionalDetails', '')}".strip()

    if "connectivity" in json_input_data and isinstance(json_input_data["connectivity"], dict):
        conn_info = json_input_data["connectivity"]
        reqs["lorawan_type_preference"] = conn_info.get("lorawanType")
        if reqs["lorawan_type_preference"] == "Private":
            reqs["special_features"]["custom_firmware"] = True
            reqs["special_features"]["manage_credentials"] = True

        if "elaborate" in conn_info and isinstance(conn_info["elaborate"], dict):
            for conn_category_key, conn_ids_list in conn_info["elaborate"].items():
                if isinstance(conn_ids_list, list):
                    for json_conn_id in conn_ids_list:
                        std_conn_term = JSON_CONNECTIVITY_MAP_GLOBAL.get(str(json_conn_id))
                        if std_conn_term: reqs["connectivity_required"].add(std_conn_term)
                        else: print(f"Warning: Unmapped JSON connectivity ID from 'elaborate.{conn_category_key}': {json_conn_id}")

    if "power" in json_input_data and isinstance(json_input_data["power"], list):
        for json_power_label in json_input_data["power"]:
            std_power_term = JSON_POWER_MAP_GLOBAL.get(str(json_power_label))
            if std_power_term: reqs["power_source_required"].add(std_power_term)
            else: print(f"Warning: Unmapped JSON power label: {json_power_label}")

    reqs["additional_details_text"] = str(json_input_data.get("additionalDetails", ""))
    if reqs["additional_details_text"]:
        reqs = nlp_extract_from_free_text_internal(reqs["additional_details_text"], reqs)

    key_set_conversion_list = ["application_subtypes_features", "connectivity_required", "power_source_required",
                               "use_case_keywords", "qualifiers", "category_specified"]
    for key_set_item in key_set_conversion_list:
        if key_set_item in reqs and isinstance(reqs[key_set_item], set):
             reqs[key_set_item] = sorted(list(reqs[key_set_item]))
    if "special_features" in reqs:
        if isinstance(reqs["special_features"].get("form_factor"), set):
            reqs["special_features"]["form_factor"] = sorted(list(reqs["special_features"]["form_factor"]))
        if isinstance(reqs["special_features"].get("power_source"), set):
            reqs["special_features"]["power_source"] = sorted(list(reqs["special_features"]["power_source"]))
    return reqs

# ==============================================================================
# --- RECOMMENDATION LOGIC (Phase 2 for JSON) ---
# ==============================================================================
def filter_products_from_json_reqs_internal(df_prods: pd.DataFrame, extracted_json_reqs: dict) -> tuple[pd.DataFrame, List[str]]:
    filtered_df = df_prods.copy(); log_messages = []

    req_env = extracted_json_reqs.get("deployment_environment")
    if req_env and 'Deployment_Environment' in filtered_df.columns:
        if str(req_env).lower() == "both":
            filtered_df = filtered_df[
                filtered_df['Deployment_Environment'].str.contains("Indoor", case=False, na=False) |
                filtered_df['Deployment_Environment'].str.contains("Outdoor", case=False, na=False) |
                filtered_df['Deployment_Environment'].str.contains("Both", case=False, na=False) # Explicit "Both"
            ]
        else:
            filtered_df = filtered_df[filtered_df['Deployment_Environment'].str.contains(req_env, case=False, na=False)]
        log_messages.append(f"Filter Deployment Env ({req_env}): {len(filtered_df)} products remaining.")
        if filtered_df.empty: return pd.DataFrame(), log_messages

    req_region = extracted_json_reqs.get("region_selected")
    req_freq_band = extracted_json_reqs.get("frequency_band_required")
    if req_region and 'Supported_Regions' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Supported_Regions'].apply(lambda x: (isinstance(x, list) and req_region in x) or (isinstance(x,list) and "Global" in x))]
        log_messages.append(f"Filter Region ({req_region}): {len(filtered_df)} products remaining.")
        if filtered_df.empty: return pd.DataFrame(), log_messages
    if req_freq_band and 'Supported_Frequency_Bands' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Supported_Frequency_Bands'].apply(lambda x: isinstance(x,list) and req_freq_band in x)]
        log_messages.append(f"Filter Freq Band ({req_freq_band}): {len(filtered_df)} products remaining.")
        if filtered_df.empty: return pd.DataFrame(), log_messages

    req_conn = set(c.lower() for c in extracted_json_reqs.get("connectivity_required", []))
    if req_conn and 'Connectivity_List' in filtered_df.columns:
        # Product must have ALL required connectivity options
        filtered_df = filtered_df[filtered_df['Connectivity_List'].apply(lambda p_list: isinstance(p_list, list) and req_conn.issubset(set(str(c).lower() for c in p_list)))]
        log_messages.append(f"Filter Required Connectivity ({req_conn}): {len(filtered_df)} products remaining.")
        if filtered_df.empty: return pd.DataFrame(), log_messages

    sf = extracted_json_reqs["special_features"]
    req_lora_ver = sf.get("lorawan_version")
    if req_lora_ver and 'Derived_LoRaWAN_Versions' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Derived_LoRaWAN_Versions'].apply(lambda v_list: isinstance(v_list, list) and req_lora_ver in v_list)]
        log_messages.append(f"Filter LoRaWAN Ver ({req_lora_ver}): {len(filtered_df)} products remaining.")
        if filtered_df.empty: return pd.DataFrame(), log_messages

    req_ip_min = sf.get("ip_rating_min")
    if req_ip_min is not None and 'Derived_IP_Rating_Numeric' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Derived_IP_Rating_Numeric'].apply(lambda ip: pd.notna(ip) and ip >= req_ip_min)]
        log_messages.append(f"Filter IP Min (IP{req_ip_min}): {len(filtered_df)} products remaining.")
        if filtered_df.empty: return pd.DataFrame(), log_messages
    
    req_cats_nlp = extracted_json_reqs.get("category_specified", [])
    if req_cats_nlp and 'Category' in filtered_df.columns:
        cat_re = '|'.join([re.escape(str(c)) for c in req_cats_nlp]) # Ensure c is string
        current_len = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Category'].str.contains(cat_re, case=False, na=False)]
        log_messages.append(f"Filter NLP Category Keywords ({req_cats_nlp}): {len(filtered_df)} products remaining (from {current_len}).")
        if filtered_df.empty: return pd.DataFrame(), log_messages
        
    return filtered_df, log_messages

def score_product_with_json_reqs_internal(
    product_row: pd.Series, product_embedding: Optional[np.ndarray],
    extracted_json_reqs: dict, query_embedding_for_add_details: Optional[np.ndarray],
    semantic_threshold: float
) -> tuple[float, List[str], float]:
    score = 0.0; details = []; H, EF, S, R, B = SCORE_HARD_REQUIREMENT_MET, SCORE_EXPLICIT_FEATURE_MATCH, SCORE_STRONG_PREFERENCE_MET, SCORE_SOFT_RELEVANCE_MET, SCORE_SEMANTIC_SIMILARITY_BOOST
    sf = extracted_json_reqs["special_features"]
    search_text_lower = str(product_row.get('Searchable_Text','')).lower()

    app_type = extracted_json_reqs.get("application_type")
    prod_cat_lower = str(product_row.get('Category','')).lower()
    if app_type:
        if prod_cat_lower and str(app_type).lower() in prod_cat_lower:
            score += S; details.append(f"AppType Category: {app_type}")
        elif str(app_type).lower() in search_text_lower :
            score += R; details.append(f"AppType Keyword: {app_type}")

    req_app_feats = set(str(f).lower() for f in extracted_json_reqs.get("application_subtypes_features", [])) # Ensure string
    prod_feat_names = set(str(fn).lower() for fn in product_row.get('Product_Feature_Names',[])) # Ensure string
    matched_app_feats = req_app_feats.intersection(prod_feat_names)
    if matched_app_feats:
        score += EF * len(matched_app_feats); details.append(f"Explicit App Features: {', '.join(sorted(list(matched_app_feats)))}")

    if extracted_json_reqs.get("connectivity_required"):
        req_conn_set = set(str(c).lower() for c in extracted_json_reqs.get("connectivity_required", [])) # Ensure string
        prod_conn_set = set(str(c).lower() for c in product_row.get('Connectivity_List', [])) # Ensure string
        if req_conn_set.issubset(prod_conn_set):
             score += H; details.append(f"All Required Connectivity Met")


    req_env = extracted_json_reqs.get("deployment_environment")
    if req_env and str(product_row.get('Deployment_Environment','')).lower() == str(req_env).lower(): # Ensure string comparison
        score += S; details.append(f"Deployment Env: {req_env}")

    req_ps = set(str(p).lower() for p in extracted_json_reqs.get("power_source_required",[])); # Ensure string
    prod_ps_derived = set(str(p).lower() for p in product_row.get('Derived_Power_Source',[])) # Ensure string
    matched_ps = req_ps.intersection(prod_ps_derived)
    if matched_ps: score += S * len(matched_ps); details.append(f"Power: {', '.join(sorted(list(matched_ps)))}")

    lora_ver = sf.get("lorawan_version")
    if lora_ver and isinstance(product_row.get('Derived_LoRaWAN_Versions'),list) and lora_ver in product_row.get('Derived_LoRaWAN_Versions',[]):
        score += S; details.append(f"LoRaWAN {lora_ver}")
    p_ip=product_row.get('Derived_IP_Rating_Numeric'); req_ip_m=sf.get("ip_rating_min")
    if pd.notna(p_ip) and req_ip_m is not None and p_ip >= req_ip_m: score+=S; details.append(f"IP{int(p_ip)} (Min IP{req_ip_m})")

    req_custom_fw = sf.get("custom_firmware"); req_manage_creds = sf.get("manage_credentials")
    prod_custom_fw = product_row.get('Derived_Custom_Firmware', False)
    if req_custom_fw and prod_custom_fw: score += S; details.append("Custom Firmware")
    if req_manage_creds and prod_custom_fw: score += S; details.append("Manages Credentials")

    uc_matched_count = 0
    for uc_kw_item in extracted_json_reqs.get("use_case_keywords", []):
        uc_kw = str(uc_kw_item).lower() # Ensure string
        if uc_kw in search_text_lower and uc_matched_count < 3:
            score += R; details.append(f"UC Keyword: {uc_kw}"); uc_matched_count+=1
            
    req_ff = set(str(f).lower() for f in sf.get("form_factor",[])); # Ensure string
    prod_ff_derived = set(str(f).lower() for f in product_row.get('Derived_Form_Factor_Keywords',[])) # Ensure string
    matched_ff = req_ff.intersection(prod_ff_derived)
    if matched_ff: score += R * len(matched_ff); details.append(f"Form Factor: {', '.join(sorted(list(matched_ff)))}")

    raw_similarity = 0.0
    if query_embedding_for_add_details is not None and product_embedding is not None and extracted_json_reqs.get("additional_details_text"):
        sim = util.cos_sim(query_embedding_for_add_details, product_embedding)[0][0].item() # type: ignore
        raw_similarity = sim
        if sim >= semantic_threshold:
            score += B * sim; details.append(f"Semantic Boost: {sim:.2f} (>{semantic_threshold:.2f})")
    return score, details, raw_similarity


def get_recommendations_from_json_input_internal(
    json_user_input_data: Dict[str, Any], df_all_products: pd.DataFrame,
    all_product_embeddings: np.ndarray, sbert_model_for_query: SentenceTransformer,
    top_n=3, current_semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:

    extracted_reqs = extract_requirements_from_json_internal(json_user_input_data)
    candidate_products_df, filter_log = filter_products_from_json_reqs_internal(df_all_products, extracted_reqs)

    if candidate_products_df.empty:
        filter_log.append("No products remained after filtering.")
        return [], extracted_reqs, filter_log

    query_text_for_semantic = extracted_reqs.get("additional_details_text", "")
    query_embedding_add_details: Optional[np.ndarray] = None
    if query_text_for_semantic and sbert_model_for_query:
        query_embedding_add_details = sbert_model_for_query.encode(query_text_for_semantic, convert_to_tensor=False)

    recs_data = []
    for original_idx, product_row in candidate_products_df.iterrows():
        product_emb = None
        if df_products_global is not None : # Ensure df_products_global is loaded
            try:
                loc = df_products_global.index.get_loc(original_idx)
                if isinstance(loc, int) and loc < len(all_product_embeddings):
                    product_emb = all_product_embeddings[loc]
                elif isinstance(loc, np.ndarray) and loc.size > 0 and loc[0] < len(all_product_embeddings): # If get_loc returns an array of positions
                     product_emb = all_product_embeddings[loc[0]]
                # else: print(f"Warning: Index {original_idx} (loc: {loc}) for product embedding lookup failed or out of bounds.")
            except KeyError:
                print(f"Warning: Index {original_idx} not found in global product data for embedding lookup.")
        else: print("Warning: df_products_global is None, cannot get embedding by original index safely.")


        final_score, matched_details, raw_similarity = score_product_with_json_reqs_internal(
            product_row, product_emb, extracted_reqs, query_embedding_add_details, current_semantic_threshold
        )
        if final_score > 0:
            recs_data.append({
                "product_id": str(product_row.get('Product_ID')),
                "product_name": str(product_row.get('Product_Name')),
                "product_model": str(product_row.get('Product_Model', '')),
                "category": str(product_row.get('Category', 'N/A')),
                "deployment_environment": str(product_row.get('Deployment_Environment', '')),
                "connectivity": ", ".join(product_row.get('Connectivity_List',[]) if isinstance(product_row.get('Connectivity_List'), list) else []),
                "product_link": str(product_row.get('Product Link', '')), # Add product link
                "score": round(final_score, 2),
                "reasoning": "; ".join(matched_details) or "General match.",
                "similarity_score_add_details": float(f"{raw_similarity:.4f}") if query_text_for_semantic else 0.0
            })
    ranked = sorted(recs_data, key=lambda x: x["score"], reverse=True)
    return ranked[:top_n], extracted_reqs, filter_log


# ==============================================================================
# --- FastAPI Application ---
# ==============================================================================
app = FastAPI(title="RAKWireless Product Recommender v2 (JSON Input)")
origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://localhost:8080", "http://localhost:8888", "*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class JsonClientInfo(BaseModel):
    name: Optional[str] = None; email: Optional[str] = None
    company: Optional[str] = None; contactNumber: Optional[str] = None
class JsonRegion(BaseModel): selected: Optional[str] = None; frequencyBand: Optional[str] = None
class JsonDeployment(BaseModel): environment: Optional[str] = None
class JsonApplication(BaseModel):
    type: Optional[str] = None; subtypes: List[str] = Field(default_factory=list)
    otherSubtype: Optional[str] = None
class JsonConnectivityElaborate(BaseModel):
    wirelessCommunication: List[str] = Field(default_factory=list); gnssGps: List[str] = Field(default_factory=list)
    wiredInterfaces: List[str] = Field(default_factory=list); protocolsDataBuses: List[str] = Field(default_factory=list)
    sensorsIO: List[str] = Field(default_factory=list)
class JsonConnectivity(BaseModel):
    lorawanType: Optional[str] = None
    elaborate: JsonConnectivityElaborate = Field(default_factory=JsonConnectivityElaborate)
class JsonQueryInput(BaseModel):
    clientInfo: Optional[JsonClientInfo] = None; region: Optional[JsonRegion] = None
    deployment: Optional[JsonDeployment] = None; application: Optional[JsonApplication] = None
    scale: Optional[str] = None; connectivity: Optional[JsonConnectivity] = None
    power: List[str] = Field(default_factory=list); additionalDetails: Optional[str] = None

class RecItemJson(BaseModel):
    product_id: str; product_name: str; product_model: Optional[str] = ""; category: Optional[str] = ""
    deployment_environment: Optional[str] = ""; connectivity: Optional[str] = ""
    product_link: Optional[str] = None # Added product_link
    score: float; reasoning: str; similarity_score_add_details: float
class ReqJsonResponse(BaseModel):
    recommendations: List[RecItemJson]; submission_id: int
    extracted_requirements: Dict[str, Any]

os.makedirs(LOGS_DIR, exist_ok=True)
SUBMISSION_LOG_FILE = os.path.join(LOGS_DIR, "submissions_json_input.jsonl")
SELECTION_LOG_FILE = os.path.join(LOGS_DIR, "selections_json_input.jsonl")

def log_to_jsonl(file_path: str, data: dict):
    data["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        with open(file_path, "a") as f: f.write(json.dumps(data, default=str) + "\n")
    except Exception as e: print(f"Error logging to {file_path}: {e}")

def needs_reprocessing(source_file_path, target_file_path):
    if not os.path.exists(target_file_path): return True
    if not os.path.exists(source_file_path): print(f"Warning: Source {source_file_path} missing for reprocessing check."); return False
    return os.path.getmtime(source_file_path) > os.path.getmtime(target_file_path)

@app.on_event("startup")
async def startup_event_tasks():
    global df_products_global, product_embeddings_global, sbert_model_global, df_features_global, nlp_global
    print("FastAPI App Startup: Initializing resources for JSON input recommender...")
    os.makedirs(DATA_DIR, exist_ok=True)
    if needs_reprocessing(SOURCE_EXCEL_FILE, PRODUCTS_PKL) or not os.path.exists(PRODUCTS_PKL):
        print(f"Preprocessing (JSON variant) Phase 0: {SOURCE_EXCEL_FILE} -> {DATA_DIR}")
        df_products_global = run_phase0_preprocessing_json_input(SOURCE_EXCEL_FILE, DATA_DIR)
    else:
        print(f"Loading preprocessed product data from {PRODUCTS_PKL}...")
        with open(PRODUCTS_PKL, "rb") as f: df_products_global = pickle.load(f)
    if df_products_global is None: raise RuntimeError("CRITICAL: df_products_global is None after Phase 0.")
    print(f"Products loaded/processed: {len(df_products_global)}")

    print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
    sbert_model_global = SentenceTransformer(SBERT_MODEL_NAME)
    print("SBERT model loaded.")
    if needs_reprocessing(PRODUCTS_PKL, EMBEDDINGS_NPY) or not os.path.exists(EMBEDDINGS_NPY):
        print(f"Regenerating Phase 1.5 Embeddings for JSON variant...")
        if df_products_global is not None and sbert_model_global is not None:
            product_embeddings_global = run_phase1_5_embeddings(df_products_global, DATA_DIR, sbert_model_global)
        else: raise RuntimeError("Cannot generate embeddings: df_products_global or sbert_model_global is None.")
    elif os.path.exists(EMBEDDINGS_NPY): product_embeddings_global = np.load(EMBEDDINGS_NPY)
    else: raise FileNotFoundError(f"Embeddings {EMBEDDINGS_NPY} not found and regeneration failed.")
    if product_embeddings_global is None: raise RuntimeError("CRITICAL: product_embeddings_global is None.")
    print(f"Embeddings loaded/generated: {len(product_embeddings_global)}")
    if len(df_products_global) != len(product_embeddings_global):
        print(f"CRITICAL WARNING: Product count ({len(df_products_global)}) / Embedding count ({len(product_embeddings_global)}) mismatch!")

    print("Loading NLP (spaCy) resources for 'additionalDetails'...")
    load_nlp_resources_internal()
    try:
        if os.path.exists(FEATURES_PKL):
            with open(FEATURES_PKL, "rb") as f: temp_df_features = pickle.load(f)
            if isinstance(temp_df_features, pd.DataFrame) and 'Feature_ID' in temp_df_features.columns:
                df_features_global = temp_df_features.set_index('Feature_ID')
            elif isinstance(temp_df_features, pd.DataFrame) and temp_df_features.index.name == 'Feature_ID':
                df_features_global = temp_df_features
            print(f"Features loaded: {len(df_features_global) if df_features_global is not None else 'None'}")
        else: print(f"Note: Features file {FEATURES_PKL} not found, df_features_global will be None.")
    except Exception as e: print(f"Note: Could not load features file {FEATURES_PKL}: {e}")
    print("Application startup sequence complete.")

@app.post("/submit-json-requirement", response_model=ReqJsonResponse)
async def api_submit_json_requirement(json_query: JsonQueryInput, request: Request):
    if not all([df_products_global is not None, product_embeddings_global is not None,
                sbert_model_global is not None, nlp_global is not None]):
        raise HTTPException(status_code=503, detail="Service not ready or not all models loaded.")
    query_dict = json_query.model_dump(exclude_none=True)
    recs, ex_reqs, f_log = get_recommendations_from_json_input_internal(
        query_dict, df_products_global, product_embeddings_global,
        sbert_model_global, top_n=5, current_semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD
    )
    sub_id = int(datetime.datetime.utcnow().timestamp() * 1000)
    serializable_ex_reqs = {}
    for k, v in ex_reqs.items():
        if isinstance(v, dict):
            serializable_ex_reqs[k] = {sk: (list(sv) if isinstance(sv, set) else sv) for sk, sv in v.items()}
            if "form_factor" in serializable_ex_reqs[k] and isinstance(serializable_ex_reqs[k].get("form_factor"), set):
                serializable_ex_reqs[k]["form_factor"] = list(serializable_ex_reqs[k]["form_factor"])
            if "power_source" in serializable_ex_reqs[k] and isinstance(serializable_ex_reqs[k].get("power_source"), set):
                serializable_ex_reqs[k]["power_source"] = list(serializable_ex_reqs[k]["power_source"])
        elif isinstance(v, set): serializable_ex_reqs[k] = list(v)
        else: serializable_ex_reqs[k] = v
    log_data = {
        "submission_id": sub_id,
        "user_info": json_query.clientInfo.model_dump() if json_query.clientInfo else {},
        "full_json_query": query_dict, "recommendations_generated": [dict(r) for r in recs],
        "extracted_requirements_log": serializable_ex_reqs, "filter_log": " ".join(f_log),
        "user_ip": request.client.host if request.client else "N/A",
        "user_agent": request.headers.get("user-agent", "N/A")
    }
    log_to_jsonl(SUBMISSION_LOG_FILE, log_data)
    return ReqJsonResponse(recommendations=recs, submission_id=sub_id, extracted_requirements=serializable_ex_reqs)

# --- LEGACY ENDPOINT AND SELECTION LOGGING (UNCHANGED FROM YOUR ORIGINAL) ---
class UserQueryLegacy(BaseModel): name: str; email: str; description: str
class RecItemLegacy(BaseModel): product_name: str; product_model: Optional[str]=""; category: Optional[str]=""; score: float; reasoning: str; similarity_score: float
class ReqResponseLegacy(BaseModel): recommendations: List[RecItemLegacy]; submission_id: int; extracted_requirements: Dict[str, Any]
@app.post("/submit-requirement-legacy", response_model=ReqResponseLegacy, deprecated=True)
async def api_submit_requirement_legacy(query: UserQueryLegacy, request: Request):
    # This is a simplified version for the legacy endpoint. Realistically, you'd maintain
    # separate NLP/scoring if its logic was significantly different.
    if not all([df_products_global is not None, product_embeddings_global is not None, sbert_model_global is not None, nlp_global is not None]):
        raise HTTPException(status_code=503, detail="Service not ready.")
    print("WARNING: /submit-requirement-legacy called. Ensure NLP resources are appropriate for free-text only.")
    temp_reqs_for_legacy = { "additional_details_text": query.description, "special_features": {}, "connectivity_required": set(), "category_specified": set(), "use_case_keywords": set(), "qualifiers": set() }
    ex_reqs_legacy = nlp_extract_from_free_text_internal(query.description, temp_reqs_for_legacy)
    recs, _, f_log = get_recommendations_from_json_input_internal(
        {"additionalDetails": query.description}, df_products_global, product_embeddings_global,
        sbert_model_global, top_n=3, current_semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD
    )
    legacy_recs_adapted = [RecItemLegacy(**r) for r in recs] # This might fail if rec structure from new func changed too much for RecItemLegacy
    sub_id = int(datetime.datetime.utcnow().timestamp() * 1000)
    log_data = {"submission_id":sub_id,"user_name":query.name,"user_email":query.email,
                "user_description":query.description,"recommendations_generated":[dict(r) for r in legacy_recs_adapted],
                "extracted_requirements_log":ex_reqs_legacy,"filter_log":" ".join(f_log),
                "user_ip":request.client.host if request.client else "N/A","user_agent":request.headers.get("user-agent","N/A")}
    log_to_jsonl(os.path.join(LOGS_DIR, "submissions_legacy.jsonl"),log_data)
    return ReqResponseLegacy(recommendations=legacy_recs_adapted,submission_id=sub_id,extracted_requirements=ex_reqs_legacy)
class SelLogItem(BaseModel): name: str; model: str; product_id: Optional[str] = None
class SelLog(BaseModel): submission_id: int; name: Optional[str] = None; email: Optional[str] = None; selected_products: List[SelLogItem]
@app.post("/log-selection")
async def api_log_selection(sel_data: SelLog, request: Request):
    log_data = {"submission_id":sel_data.submission_id,"user_name":sel_data.name,"user_email":sel_data.email,
                "selected_products":[item.model_dump() for item in sel_data.selected_products],
                "user_ip":request.client.host if request.client else "N/A","user_agent":request.headers.get("user-agent","N/A")}
    log_to_jsonl(SELECTION_LOG_FILE,log_data)
    return {"status":"selection logged","redirect_url":STORE_URL}
# --- END LEGACY ---

if __name__ == "__main__":
    print("Starting main.py for local testing (not FastAPI server)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    df_prods = run_phase0_preprocessing_json_input(SOURCE_EXCEL_FILE, DATA_DIR)
    print(f"Phase 0 processed {len(df_prods)} products.")
    print(df_prods.head()[['Product_ID', 'Product_Name', 'Deployment_Environment', 'Connectivity_List', 'Derived_Power_Source', 'Derived_IP_Rating_Numeric', 'Supported_Frequency_Bands']].to_string())

    sbert_test_model = SentenceTransformer(SBERT_MODEL_NAME)
    embeddings_arr_test = run_phase1_5_embeddings(df_prods, DATA_DIR, sbert_test_model)
    print(f"Generated {len(embeddings_arr_test)} embeddings.")

    load_nlp_resources_internal()
    print("NLP resources loaded for local test.")

    sample_json_test_input = {
      "clientInfo": {"name": "Local Test User"},
      "region": {"selected": "India", "frequencyBand": "IN865"},
      "deployment": {"environment": "Outdoor"},
      "application": {"type": "Environmental Monitoring", "subtypes": ["Air Quality", "Temperature"], "otherSubtype": ""},
      "connectivity": {
        "lorawanType": "Public",
        "elaborate": {"wirelessCommunication": ["lorawan_protocol", "ble"], "wiredInterfaces": ["ethernet_wired"]}
      },
      "power": ["Solar Power", "Battery Powered"],
      "additionalDetails": "Needs to be very compact and reliable. IP67 is a must. LoRaWAN 1.0.3 support required."
    }
    print(f"\n--- Test Recommendation for Sample JSON ---")
    print(f"Input: {json.dumps(sample_json_test_input, indent=2)}")
    
    test_recs, test_ex_reqs, test_filter_log = get_recommendations_from_json_input_internal(
        sample_json_test_input, df_prods, embeddings_arr_test, sbert_test_model, top_n=3
    )

    print("\nExtracted Requirements (Local Test):")
    printable_test_ex_reqs = {}
    for k,v in test_ex_reqs.items():
        if isinstance(v, dict):
            printable_test_ex_reqs[k] = {sk: (list(sv) if isinstance(sv, set) else sv) for sk, sv in v.items()}
        elif isinstance(v, set): printable_test_ex_reqs[k] = list(v)
        else: printable_test_ex_reqs[k] = v
    print(json.dumps(printable_test_ex_reqs, indent=2, default=str))
    
    print("\nFilter Log (Local Test):")
    for log_entry in test_filter_log: print(f"  {log_entry}")

    if test_recs:
        print("\nTop Recommendations (Local Test):")
        for i, rec in enumerate(test_recs):
            print(f"  Rank {i+1}: {rec['product_name']} (ID: {rec.get('product_id')}, Model: {rec['product_model']}) Score: {rec['score']}")
            print(f"    Reason: {rec['reasoning']}")
            print(f"    Link: {rec.get('product_link')}")
            print(f"    Similarity (Add.Details): {rec.get('similarity_score_add_details', 0.0):.4f}")
    else:
        print("No recommendations found for local test input.")