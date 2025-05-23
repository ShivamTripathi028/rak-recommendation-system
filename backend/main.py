# backend/main.py (NEW - Based on "Older Logic" and compatible with current frontend)

# --- Standard Library Imports ---
import os
import json
import re
import unicodedata
from typing import List, Dict, Any, Optional, Set

# --- Third-Party Imports ---
import pandas as pd
from fastapi import FastAPI, HTTPException, Request # Added Request for potential future use (e.g., IP logging)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

# --- Application Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# *** IMPORTANT: Update this if your Excel filename is different ***
DATASET_EXCEL_FILE = os.path.join(BASE_DIR, "iot_products_dataset.xlsx")
# If your Excel file is named "RAKWireless Recommender Dataset.xlsx" (as in your reference structure), use:
# DATASET_EXCEL_FILE = os.path.join(BASE_DIR, "RAKWireless Recommender Dataset.xlsx")


# --- Constants for Recommender ---
FUZZY_MATCH_THRESHOLD = 75
PARTIAL_FUZZY_MATCH_THRESHOLD = 85

# --- Helper Functions (normalize_text, slugify) ---
def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""
    if not text:
        return ""
    try:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    except Exception:
        # Fallback if normalization fails for any reason
        pass
    # Convert to lowercase and strip extra whitespace
    return re.sub(r'\s+', ' ', text.lower()).strip()

def slugify(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else "na" # Ensure it's a string, default to "na"
    text = normalize_text(text)
    if not text: # if normalization results in empty string
        return "na"
    text = text.replace('_', '-') # Replace underscores with hyphens first
    text = re.sub(r'[^\w\s-]', '', text)  # Remove non-alphanumeric chars (except whitespace, hyphen)
    text = re.sub(r'\s+', '-', text)  # Replace spaces with hyphens
    text = re.sub(r'[^a-z0-9\-]', '', text) # Ensure only lowercase alphanumeric and hyphens
    text = re.sub(r'\-{2,}', '-', text)  # Replace multiple hyphens with single
    text = text.strip('-')
    return text if text else "na" # Final check for empty string after slugification

# --- Mappings (from JSON frontend values to searchable keywords/concepts) ---
CONNECTIVITY_JSON_TO_KEYWORDS: Dict[str, List[str]] = {
    # Wireless Communication
    "lorawan": ["lorawan", "lorawan protocol"], # from your frontend enum: "lorawan"
    "lora_p2p": ["lora p2p", "lora point-to-point"],
    "lora": ["lora"], # "lora" from enum
    "meshtastic": ["meshtastic", "mesh"],
    "wifi": ["wifi", "wi-fi", "wlan", "802.11"],
    "bluetooth": ["bluetooth classic", "bluetooth"], # "bluetooth" from enum
    "ble": ["ble", "bluetooth low energy"],
    "nfc": ["nfc", "near field communication"],
    "uwb": ["uwb", "ultra-wideband"],
    "lte": ["lte", "4g", "5g", "cellular"], # "lte" from enum
    "lte_m_cat_m1": ["lte-m", "ltem", "cat-m1", "cat m1"], # "lte_m_cat_m1" from enum
    "nb_iot": ["nb-iot", "nbiot"],
    "gsm": ["gsm", "2g"],
    "agw": ["agw", "aggregation gateway"],
    "lpwan_other": ["lpwan"], # "lpwan_other" from enum
    # GNSS / GPS
    "gps": ["gps"], # "gps" from enum
    "gnss": ["gnss", "glonass", "beidou", "galileo", "qzss"], # "gnss" from enum
    # Wired Interfaces
    "ethernet": ["ethernet", "rj45"], # "ethernet" from enum
    "poe": ["poe", "power over ethernet"],
    "usb": ["usb", "usb-c", "usb 2.0", "usb 3.0"], # "usb" from enum
    "pcie": ["pcie", "pci express"],
    "twisted_pair": ["twisted pair"],
    "coaxial_cable": ["coaxial cable"],
    # Protocols / Data Buses
    "i2c": ["i2c", "i²c"], # "i2c" from enum
    "spi": ["spi"], # "spi" from enum
    "uart_serial": ["uart", "serial", "rs232"],
    "rs485": ["rs485", "rs-485"],
    "sdi12": ["sdi-12", "sdi12"],
    "can_bus": ["can", "can bus"],
    "lin_bus": ["lin", "lin bus"],
    "mqtt": ["mqtt"],
    # Sensors / IO
    "adc": ["adc", "analog to digital"], # "adc" from enum
    "digital_io": ["digital i/o", "dio", "digital input output", "gpio"], # "digital_io" from enum
}

POWER_JSON_TO_KEYWORDS: Dict[str, List[str]] = {
    # Frontend values are already descriptive. We normalize them for the keys here.
    # The value in the list is what we search for in the product text.
    "dc power": ["dc power", "direct current", "dc input"],
    "ac power": ["ac power", "alternating current", "ac input", "mains power"],
    "battery powered": ["battery", "battery powered", "battery-operated"],
    "usb powered": ["usb power", "usb powered", "powered by usb"],
    "solar power": ["solar power", "solar panel", "photovoltaic"],
    "poe (power over ethernet)": ["poe", "power over ethernet"],
    "other / not specified": [] # This option means no specific power keywords to search
}


# --- IoTProductRecommender Class (Our "Older Logic" Recommender) ---
class IoTProductRecommender:
    def __init__(self, excel_dataset_path: str):
        self.dataset_path = excel_dataset_path
        self.products_df = self._load_and_preprocess_dataset(sheet_name="Product Table")
        self.weights = {
            "frequency_band": 0.25, "environment": 0.15, "application": 0.20,
            "connectivity": 0.20, "power": 0.10, "additional_details": 0.10
        }

    def _load_and_preprocess_dataset(self, sheet_name: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(self.dataset_path, sheet_name=sheet_name)
        except FileNotFoundError:
            print(f"ERROR: Dataset file not found at {self.dataset_path}")
            return pd.DataFrame()
        except ValueError as ve:
            if "Worksheet named" in str(ve) and sheet_name in str(ve):
                 print(f"ERROR: Worksheet named '{sheet_name}' not found in {self.dataset_path}.")
            else:
                 print(f"ERROR: Could not load Excel file/sheet {self.dataset_path} (Sheet: {sheet_name}): {ve}")
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR: Could not load Excel file {self.dataset_path} (Sheet: {sheet_name}): {e}")
            return pd.DataFrame()

        required_columns = [
            'Product_Name', 'Product_Model', 'Description_And_Application',
            'Connectivity', 'Region Support', 'Notes', 'Deployment_Environment'
            # 'Product_Line', 'Product_Subline', 'Hardware/Software' are desirable but less critical for matching
        ]
        self.has_product_id_col = 'Product_ID' in df.columns
        
        for col in required_columns:
            if col not in df.columns:
                print(f"ERROR: Critical column '{col}' is missing from '{sheet_name}'. Cannot proceed with recommendation logic.")
                return pd.DataFrame() # Return empty if critical column is missing

        # Store original casing for display, then normalize for search
        df['Product_Name_Original'] = df['Product_Name'].astype(str)
        df['Product_Model_Original'] = df['Product_Model'].astype(str)

        for col_to_normalize in ['Product_Name', 'Product_Model', 'Description_And_Application', 'Notes', 'Deployment_Environment']:
            if col_to_normalize in df.columns:
                 df[col_to_normalize] = df[col_to_normalize].apply(normalize_text)
            else: # Should not happen for critical columns due to check above
                 df[col_to_normalize] = ""

        if 'Region Support' in df.columns:
            df['Region_Support_list_normalized'] = df['Region Support'].apply(
                lambda x: [normalize_text(i.strip()) for i in str(x).split(',')] if pd.notna(x) and str(x).strip() else []
            )
        else: df['Region_Support_list_normalized'] = pd.Series([[] for _ in range(len(df))]) # Should be caught by critical check

        if 'Connectivity' in df.columns:
            df['Connectivity_raw_normalized'] = df['Connectivity'].apply(normalize_text)
            df['Connectivity_keywords_set'] = df['Connectivity'].apply(
                lambda x: set(normalize_text(i.strip()) for i in str(x).split(',')) if pd.notna(x) and str(x).strip() else set()
            )
        else: # Should be caught by critical check
            df['Connectivity_raw_normalized'] = ""
            df['Connectivity_keywords_set'] = pd.Series([set() for _ in range(len(df))])

        df['combined_description_notes'] = df['Description_And_Application'] + " " + df['Notes']
        print(f"Successfully loaded and preprocessed '{sheet_name}' sheet. {len(df)} products.")
        return df

    def _get_client_preferences(self, client_json_input_fields: Dict[str, Any]) -> Dict[str, Any]:
        prefs: Dict[str, Any] = {}

        region_info = client_json_input_fields.get("region", {})
        prefs['frequency_band'] = normalize_text(region_info.get("frequencyBand", ""))
        
        deployment_info = client_json_input_fields.get("deployment", {})
        prefs['environment'] = normalize_text(deployment_info.get("environment", ""))

        app_info = client_json_input_fields.get("application", {})
        prefs['app_type'] = normalize_text(app_info.get("type", ""))
        raw_subtypes = app_info.get("subtypes", [])
        prefs['app_subtypes'] = [normalize_text(st) for st in raw_subtypes if isinstance(st, str)]
        
        other_subtype_text = normalize_text(app_info.get("otherSubtype", ""))
        if "other" in prefs['app_subtypes'] and other_subtype_text:
            prefs['app_subtypes'].append(other_subtype_text) # Add the specific "other" text
            prefs['app_subtypes'] = [st for st in prefs['app_subtypes'] if st != "other"] # Remove placeholder "Other"
        
        conn_info = client_json_input_fields.get("connectivity", {})
        elaborate_conn = conn_info.get("elaborate", {})
        selected_connectivity_ids: List[str] = []
        if isinstance(elaborate_conn, dict):
            for category_values in elaborate_conn.values(): # e.g., elaborate_conn["wirelessCommunication"]
                if isinstance(category_values, list):
                    selected_connectivity_ids.extend([normalize_text(opt) for opt in category_values if isinstance(opt, str)])
        prefs['connectivity_options_ids'] = selected_connectivity_ids
        
        power_info = client_json_input_fields.get("power", [])
        prefs['power_options_labels'] = [normalize_text(p) for p in power_info if isinstance(p, str)]

        prefs['additional_details'] = normalize_text(client_json_input_fields.get("additionalDetails", ""))
        return prefs

    def _score_frequency_band(self, product_region_support_list: List[str], required_freq_band: str) -> float:
        if not required_freq_band: return 0.5
        if not product_region_support_list: return 0.0
        for region_item in product_region_support_list:
            if required_freq_band == region_item or fuzz.ratio(required_freq_band, region_item) > 90:
                return 1.0
        return 0.0

    def _score_environment(self, product_env_str: str, required_env_str: str) -> float:
        if not required_env_str: return 0.5
        if not product_env_str and required_env_str : return 0.1

        if required_env_str == "both":
            if product_env_str == "both": return 1.0
            if product_env_str in ["indoor", "outdoor"]: return 0.8
        elif required_env_str == "indoor":
            if product_env_str == "indoor" or product_env_str == "both": return 1.0
        elif required_env_str == "outdoor":
            if product_env_str == "outdoor" or product_env_str == "both": return 1.0
        if required_env_str in product_env_str: return 0.6
        return 0.1

    def _score_application(self, product_desc_app_str: str, app_type: str, app_subtypes: List[str]) -> float:
        if not app_type and not app_subtypes: return 0.5
        search_terms = [term for term in ([app_type] + app_subtypes) if term]
        if not search_terms: return 0.5
        if not product_desc_app_str: return 0.0

        total_score = 0.0
        if app_type and fuzz.partial_ratio(app_type, product_desc_app_str, score_cutoff=PARTIAL_FUZZY_MATCH_THRESHOLD) > 0:
            total_score += 0.5
        
        matched_subtype_count = 0
        for subtype in app_subtypes:
            if fuzz.partial_ratio(subtype, product_desc_app_str, score_cutoff=PARTIAL_FUZZY_MATCH_THRESHOLD) > 0:
                matched_subtype_count +=1
        
        if app_subtypes:
            total_score += (matched_subtype_count / len(app_subtypes)) * 0.5
        elif app_type and not app_subtypes and total_score > 0:
            return 1.0
        return min(total_score, 1.0)

    def _score_connectivity(self, product_conn_keywords_set: Set[str], product_conn_raw_str: str,
                            product_combined_desc_notes: str, client_conn_ids_list: List[str]) -> float:
        if not client_conn_ids_list: return 0.5
        
        num_client_options = len(client_conn_ids_list)
        matched_options = 0

        for conn_id in client_conn_ids_list:
            search_keywords_for_id = CONNECTIVITY_JSON_TO_KEYWORDS.get(conn_id, [conn_id])
            option_matched_in_product = False
            for keyword in search_keywords_for_id:
                if keyword in product_conn_keywords_set: option_matched_in_product = True; break
                if not option_matched_in_product and product_conn_raw_str and \
                   fuzz.partial_ratio(keyword, product_conn_raw_str, score_cutoff=PARTIAL_FUZZY_MATCH_THRESHOLD) > 0:
                    option_matched_in_product = True; break
                if not option_matched_in_product and \
                   fuzz.partial_ratio(keyword, product_combined_desc_notes, score_cutoff=PARTIAL_FUZZY_MATCH_THRESHOLD) > 0:
                    option_matched_in_product = True; break
            if option_matched_in_product: matched_options += 1
        return (matched_options / num_client_options) if num_client_options > 0 else 0.0

    def _score_power(self, product_combined_desc_notes: str, client_power_labels_list: List[str]) -> float:
        if not client_power_labels_list or \
           ("other / not specified" in client_power_labels_list and len(client_power_labels_list) == 1):
            return 0.5
        actual_power_prefs = [p for p in client_power_labels_list if p != "other / not specified"]
        if not actual_power_prefs: return 0.5
        if not product_combined_desc_notes: return 0.0

        matched_power_options = 0
        for power_label in actual_power_prefs:
            search_keywords_for_label = POWER_JSON_TO_KEYWORDS.get(power_label, [power_label])
            if any(fuzz.partial_ratio(keyword, product_combined_desc_notes, score_cutoff=PARTIAL_FUZZY_MATCH_THRESHOLD) > 0 for keyword in search_keywords_for_label):
                matched_power_options += 1
        return (matched_power_options / len(actual_power_prefs)) if actual_power_prefs else 0.0

    def _score_additional_details(self, product_combined_desc_notes: str, additional_details_text: str) -> float:
        if not additional_details_text: return 0.5
        if not product_combined_desc_notes: return 0.0

        specific_keywords = ["rui3", "ip67", "ip65", "ip54", "sdk", "at command", "mqtt", "spi", "custom firmware"]
        keyword_bonus = 0.0; found_specific = 0
        for skey in specific_keywords:
            if skey in additional_details_text and fuzz.partial_ratio(skey, product_combined_desc_notes, score_cutoff=90) > 0:
                 found_specific +=1
        if found_specific > 0: keyword_bonus = (found_specific / len(specific_keywords)) * 0.3
        fuzzy_score = fuzz.partial_token_set_ratio(additional_details_text, product_combined_desc_notes, score_cutoff=FUZZY_MATCH_THRESHOLD) / 100.0
        return min(fuzzy_score * 0.7 + keyword_bonus, 1.0)

    def recommend(self, client_json_input_fields: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
        if self.products_df.empty:
            print("ERROR: Product DataFrame is empty in recommend method.")
            return []
        try:
            prefs = self._get_client_preferences(client_json_input_fields)
        except Exception as e:
            print(f"ERROR parsing preferences: {e}")
            return []

        recommendations = []
        for _, product_row in self.products_df.iterrows():
            s_freq = self._score_frequency_band(product_row['Region_Support_list_normalized'], prefs['frequency_band'])
            s_env = self._score_environment(product_row['Deployment_Environment'], prefs['environment'])
            s_app = self._score_application(product_row['Description_And_Application'], prefs['app_type'], prefs['app_subtypes'])
            s_conn = self._score_connectivity(
                product_row['Connectivity_keywords_set'], product_row['Connectivity_raw_normalized'],
                product_row['combined_description_notes'], prefs['connectivity_options_ids']
            )
            s_power = self._score_power(product_row['combined_description_notes'], prefs['power_options_labels'])
            s_add_details = self._score_additional_details(product_row['combined_description_notes'], prefs['additional_details'])

            current_score = (
                s_freq * self.weights['frequency_band'] + s_env * self.weights['environment'] +
                s_app * self.weights['application'] + s_conn * self.weights['connectivity'] +
                s_power * self.weights['power'] + s_add_details * self.weights['additional_details']
            )
            
            prod_id_val = product_row.get('Product_ID') if self.has_product_id_col else None
            product_id_for_url = slugify(prod_id_val if pd.notna(prod_id_val) and str(prod_id_val).strip() else product_row['Product_Model_Original'])
            
            recommendations.append({
                'product_id': product_id_for_url,
                'product_name': product_row['Product_Name_Original'],
                'product_model': product_row['Product_Model_Original'],
                'score': round(current_score, 4),
            })
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]

# --- FastAPI Application Setup ---
app = FastAPI(title="RAK IoT Product Recommender")
origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://localhost:8080", "http://localhost:8888",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["POST", "OPTIONS"], allow_headers=["Content-Type"],
)

# --- Pydantic Models for Request Body Validation ---
class ClientInfo(BaseModel): name: str; email: str; company: Optional[str] = None; contactNumber: str
class RegionInfo(BaseModel): selected: str; frequencyBand: str
class DeploymentInfo(BaseModel): environment: Optional[str] = None
class ApplicationInfo(BaseModel): type: str; subtypes: List[str]; otherSubtype: Optional[str] = None
class ElaborateConnectivity(BaseModel):
    wirelessCommunication: List[str] = Field(default_factory=list); gnssGps: List[str] = Field(default_factory=list)
    wiredInterfaces: List[str] = Field(default_factory=list); protocolsDataBuses: List[str] = Field(default_factory=list)
    sensorsIO: List[str] = Field(default_factory=list)
class ConnectivityInfo(BaseModel): elaborate: ElaborateConnectivity
class FormFields(BaseModel):
    clientInfo: ClientInfo; region: RegionInfo; deployment: DeploymentInfo; application: ApplicationInfo
    scale: str; connectivity: ConnectivityInfo; power: List[str]; additionalDetails: Optional[str] = None
class FullQueryInput(BaseModel): fields: FormFields; formSchemaDescription: Optional[str] = None

recommender_instance: Optional[IoTProductRecommender] = None
initialization_error_message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global recommender_instance, initialization_error_message
    print("FastAPI App Startup: Initializing IoTProductRecommender...")
    try:
        if not os.path.exists(DATASET_EXCEL_FILE):
            initialization_error_message = f"CRITICAL: Dataset Excel file not found at {DATASET_EXCEL_FILE}"
            print(initialization_error_message); return
        recommender_instance = IoTProductRecommender(excel_dataset_path=DATASET_EXCEL_FILE)
        if recommender_instance.products_df.empty:
            initialization_error_message = "CRITICAL: Product DataFrame is empty after loading. Check Excel file content and 'Product Table' sheet."
            print(initialization_error_message); recommender_instance = None
        else:
            print(f"Recommender initialized successfully. Loaded {len(recommender_instance.products_df)} products from 'Product Table'.")
    except Exception as e:
        initialization_error_message = f"CRITICAL: Error during recommender initialization: {e}"
        print(initialization_error_message); recommender_instance = None

@app.post("/submit-json-requirement")
async def get_recommendations_api(query: FullQueryInput): # Removed Request from params as it's not used
    if initialization_error_message or recommender_instance is None:
        print(f"Service unavailable: {initialization_error_message}")
        raise HTTPException(status_code=503, detail=f"Recommendation service unavailable: {initialization_error_message}")
    
    recommendations = recommender_instance.recommend(query.fields.model_dump(exclude_none=True, by_alias=False), top_n=3)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    print(f"Attempting to initialize recommender with dataset: {DATASET_EXCEL_FILE}, Sheet: 'Product Table'")
    if not os.path.exists(DATASET_EXCEL_FILE):
        print(f"ERROR: Dataset file '{DATASET_EXCEL_FILE}' not found. Ensure it's in the same directory as main.py or update path.")
    else:
        test_recommender = IoTProductRecommender(DATASET_EXCEL_FILE)
        if test_recommender.products_df.empty:
            print("Test recommender loaded an empty dataset. Check Excel file or logs for errors.")
        else:
            print(f"Test recommender initialized with {len(test_recommender.products_df)} products from 'Product Table'.")
            sample_form_fields_data = {
                "clientInfo": {"name": "Local Test User", "email": "local@test.com", "contactNumber": "555-0123"},
                "region": {"selected": "Europe (868 MHz)", "frequencyBand": "EU868"},
                "deployment": {"environment": "Outdoor"},
                "application": {"type": "Asset Tracking", "subtypes": ["GPS Tracking", "Vibration"], "otherSubtype": ""},
                "scale": "Medium Deployment (11-50 devices)",
                "connectivity": {"elaborate": {"wirelessCommunication": ["lorawan", "ble"], "gnssGps": ["gps"]}},
                "power": ["Battery Powered", "Solar Power"],
                "additionalDetails": "Device must be rugged, IP67, and support custom firmware for LoRaWAN. Low power is key."
            }
            print("\n--- Local Test with Sample Form Fields Data ---")
            test_recs = test_recommender.recommend(sample_form_fields_data, top_n=3)
            if test_recs:
                print("Recommendations:")
                for i, rec in enumerate(test_recs):
                    print(f"  Rank {i+1}: {rec['product_name']} (Model: {rec['product_model']}), Score: {rec['score']:.4f}, ID: {rec['product_id']}")
            else:
                print("No recommendations found for sample data.")
    print("\nTo run the FastAPI server for frontend interaction, use from the 'backend' directory:")
    print("uvicorn main:app --reload --port 8000")