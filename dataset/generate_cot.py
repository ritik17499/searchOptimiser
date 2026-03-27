import json
import time
import os
import random
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm

# --- Configuration ---
INPUT_JSON = 'cleaned_appliances_mapped.json'
OUTPUT_JSON = 'appliances_cot_ready.json'
MAX_ITEMS_TO_PROCESS = 1000 # Limit to a 1K subset for our research proof-of-concept

API_KEY = '**'
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please export it.")

# Initialize the GenAI Client
client = genai.Client(api_key=API_KEY)

class CoTResponse(BaseModel):
    cot_core: str
    cot_visual: str
    cot_intent: str

SYSTEM_PROMPT = """
You are an e-commerce data extraction AI. Read the user review and extract the core product features.
1. "cot_core": The core item being reviewed (e.g., "blender", "hair dryer", "mini fridge").
2. "cot_visual": Any visual or physical attributes mentioned (e.g., "stainless steel", "small footprint", "black"). If none, infer a likely generic visual attribute based on the item.
3. "cot_intent": The functional intent or use case (e.g., "making smoothies", "travel", "dorm room").

Review: "{review_text}"
"""

def generate_cot_pipeline():
    print(f"Loading cleaned dataset from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Slicing dataset to the first {MAX_ITEMS_TO_PROCESS} items...")
    data_subset = {k: data[k] for k in list(data.keys())[:MAX_ITEMS_TO_PROCESS]}

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        print(f"Resuming pipeline... Found {len(final_data)} previously processed items.")
    else:
        final_data = {}
        print("Starting fresh CoT generation pipeline...")

    # Iterate through our 1K subset
    for item_id, item_data in tqdm(data_subset.items(), desc="Generating CoT"):
        
        if item_id in final_data:
            continue

        raw_text = item_data.get('text', '')
        prompt = SYSTEM_PROMPT.format(review_text=raw_text)

        max_retries = 6
        base_delay = 5 

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CoTResponse,
                        temperature=0.2 
                    ),
                )
                
                cot_json = json.loads(response.text)
                
                final_data[item_id] = {
                    **item_data, 
                    "cot_core": cot_json.get("cot_core", "appliance"),
                    "cot_visual": cot_json.get("cot_visual", "standard design"),
                    "cot_intent": cot_json.get("cot_intent", "general use")
                }

                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, indent=4)
                    
                time.sleep(6.5) 
                break 

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Resource" in error_msg or "quota" in error_msg.lower():
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    
                    print(f"\n[Rate Limit] Pausing for {sleep_time:.1f}s. Exact API Error: {error_msg}")
                    time.sleep(sleep_time)
                else:
                    print(f"\n[Error] Skipping ID {item_id} due to unexpected error: {error_msg}")
                    break 
        else:
            print(f"\n[Skipped] ID {item_id} permanently failed after {max_retries} retries.")

    print("\nPipeline complete! The 1K research subset CoT dataset is ready.")

if __name__ == "__main__":
    generate_cot_pipeline()
