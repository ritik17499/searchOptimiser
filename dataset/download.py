import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
INPUT_JSONL = 'Appliances.jsonl'  
OUTPUT_JSON = 'cleaned_appliances_mapped.json'
IMAGE_DIR = 'downloaded_images'
MAX_WORKERS = 20 # Number of concurrent downloads (adjust based on your internet speed)

# Create the image directory if it doesn't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(url, image_path):
    """Helper function to download an image and save it to disk."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        return False

def process_dataset():
    # Dictionary to hold our final clean data
    final_mapped_data = {}
    
    # We will collect valid tasks here first to feed into the thread pool
    download_tasks = []
    current_id = 1
    
    print("Step 1: Parsing JSONL and filtering out reviews without images...")
    with open(INPUT_JSONL, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                item = json.loads(line)
                
                # Filter: We only care if 'images' exists, is not empty, and has text
                if item.get('images') and len(item['images']) > 0 and item.get('text'):
                    
                    # Prefer the large image, fallback to medium or small if missing
                    img_data = item['images'][0]
                    img_url = img_data.get('large_image_url') or img_data.get('medium_image_url') or img_data.get('small_image_url')
                    
                    if img_url:
                        # Add to our task list for downloading
                        download_tasks.append({
                            'id': str(current_id),
                            'url': img_url,
                            'text': item['text'],
                            'title': item.get('title', ''),
                            'asin': item.get('asin', '')
                        })
                        current_id += 1
                        
            except json.JSONDecodeError:
                continue 

    print(f"Found {len(download_tasks)} valid reviews with images. Starting downloads...")

    # Step 2: Download images concurrently
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map futures to their specific task data so we know which one finished
        future_to_task = {
            executor.submit(download_image, task['url'], os.path.join(IMAGE_DIR, f"{task['id']}.jpg")): task 
            for task in download_tasks
        }
        
        # Use tqdm to show a progress bar
        for future in tqdm(as_completed(future_to_task), total=len(download_tasks), desc="Downloading"):
            task = future_to_task[future]
            try:
                success = future.result()
                if success:
                    # Only add to our final JSON if the image successfully downloaded
                    final_mapped_data[task['id']] = {
                        "text": task['text'],
                        "title": task['title'],
                        "asin": task['asin']
                    }
                    success_count += 1
            except Exception as e:
                pass # Skip if network dropped or connection closed

    # Step 3: Save the finalized mapping JSON
    print(f"\nSuccessfully downloaded {success_count} images.")
    print(f"Saving mapped metadata to {OUTPUT_JSON}...")
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_mapped_data, f, indent=4)
        
    print("Done! Dataset prep complete.")

if __name__ == "__main__":
    process_dataset()
