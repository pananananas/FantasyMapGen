from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import re

OUTPUT_DIR = "data/final_maps"

def get_image_groups():
    """
    Group images by their timestamp and type (base map, map with buildings, and fantasy variations)
    Returns a dictionary of format {timestamp: {type: image_path}}
    """
    all_results = {}
    
    # Get all relevant images
    all_images = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    
    for image in all_images:
        # Extract timestamp (format: YYYY-MM-DD_HH-MM-SS)
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_([^\.]+)\.png', image)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            img_type = timestamp_match.group(2)
            
            if timestamp not in all_results:
                all_results[timestamp] = {
                    'base_map': None,
                    'map_with_buildings': None,
                    'fantasy_variations': []
                }
            
            full_path = os.path.join(OUTPUT_DIR, image)
            if img_type == 'base_map':
                all_results[timestamp]['base_map'] = full_path
            elif img_type == 'map_with_buildings':
                all_results[timestamp]['map_with_buildings'] = full_path
            elif 'fantasy' in img_type:
                all_results[timestamp]['fantasy_variations'].append(full_path)
    
    # Sort fantasy variations and filter incomplete sets
    complete_results = {}
    for timestamp, data in all_results.items():
        if data['base_map'] and data['map_with_buildings'] and data['fantasy_variations']:
            data['fantasy_variations'].sort()
            complete_results[timestamp] = data
    
    return complete_results

def create_results_subplot(all_results):
    """
    Create a subplot with all generated images including base maps
    all_results: dict with structure {timestamp: {'base_map': path, 'map_with_buildings': path, 'fantasy_variations': [paths]}}
    """
    num_locations = len(all_results)
    if num_locations == 0:
        print("No complete image sets found in the output directory")
        return
    
    # Get number of fantasy variations (assuming consistent across all timestamps)
    first_result = next(iter(all_results.values()))
    num_variations = len(first_result['fantasy_variations'])
    num_images_per_row = num_variations + 2  # +2 for base map and map with buildings
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 4 * num_locations))
    fig.suptitle('Generated Fantasy Maps by Location and Variation', fontsize=16)
    
    from tqdm import tqdm  # Import tqdm for progress bar
    for idx, (timestamp, image_data) in enumerate(tqdm(all_results.items(), desc="Processing images")):
        # Plot base map
        plot_idx = idx * num_images_per_row + 1
        ax = fig.add_subplot(num_locations, num_images_per_row, plot_idx)
        img = Image.open(image_data['base_map'])
        ax.imshow(np.array(img))
        if idx == 0:
            ax.set_title('Base Map')
        ax.axis('off')
        
        # Plot map with buildings
        plot_idx = idx * num_images_per_row + 2
        ax = fig.add_subplot(num_locations, num_images_per_row, plot_idx)
        img = Image.open(image_data['map_with_buildings'])
        ax.imshow(np.array(img))
        if idx == 0:
            ax.set_title('Map with Buildings')
        ax.axis('off')
        
        # Plot fantasy variations
        for var_idx, img_path in enumerate(image_data['fantasy_variations']):
            plot_idx = idx * num_images_per_row + var_idx + 3
            ax = fig.add_subplot(num_locations, num_images_per_row, plot_idx)
            img = Image.open(img_path)
            ax.imshow(np.array(img))
            if idx == 0:
                ax.set_title(f'Fantasy Variation {var_idx + 1}')
            ax.axis('off')
        
        # Add timestamp label to the first image in the row
        if idx == 0:
            ax = plt.subplot(num_locations, 1, idx + 1)
            ax.text(-0.1, 0.5, timestamp, rotation=0, 
                   horizontalalignment='right', verticalalignment='center',
                   transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = f"{OUTPUT_DIR}/{timestamp}_results_grid.png"
    plt.savefig(results_path, bbox_inches='tight', dpi=300)
    print(f"Results grid saved to: {results_path}")
    plt.close()

def main():
    print("Reading images from:", OUTPUT_DIR)
    all_results = get_image_groups()
    print(f"Found {len(all_results)} complete location sets")
    create_results_subplot(all_results)

if __name__ == "__main__":
    main()