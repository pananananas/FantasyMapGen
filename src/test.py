from PIL import Image
from datetime import datetime
from mflux import Flux1, Config
from diffusers.utils import load_image
from diffusers import FluxImg2ImgPipeline
from draw_map import BuildingOverlay, MapColorPalettes, OSMMapGenerator
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np
import requests
import random
import torch
import os

# Configuration
USE_MLX = False 
OUTPUT_DIR = "data/final_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def run_openai_api(prompt):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    return response.choices[0].message.content

# def run_clarin_api(prompt):
#     url = "https://services.clarin-pl.eu/api/v1/oapi/chat/completions"
#     user_token = os.getenv('CLARIN_API_KEY')

#     headers = {
#         "accept": "application/json",
#         "Authorization": f"Bearer {user_token}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "llama",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     }

#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code != 200:
#         print(f"Error: {response.status_code}")
#         print(response.text)
#         return None
#     return response.json()['choices'][0]['message']['content']

def load_model(device):
    """Load the model"""
    print("Loading model...")
    if USE_MLX:
        model_config = Flux1.from_alias("dev").model_config
        pipe = Flux1(
            model_config=model_config,
            quantize=4,
            lora_paths=["loras/lora_dev_10000.safetensors"],
            lora_scales=[1.0],
        )
    else:
        HUGGINGFACE_HUB_CACHE="${SCRATCH}/huggingface_cache"
        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16,
            cache_dir=HUGGINGFACE_HUB_CACHE,
            use_safetensors=True,
        )
        pipe.enable_attention_slicing()
        pipe = pipe.to(device)
    return pipe

def generate_fantasy_map(bbox, color_palette, model, params):
    """Generate fantasy map from given coordinates and parameters"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_map_path = f"{OUTPUT_DIR}/{timestamp}_base_map.png"
    map_with_buildings_path = f"{OUTPUT_DIR}/{timestamp}_map_with_buildings.png"
    house_icon_path = "icons/house.png"

    south, west = bbox['bottom'], bbox['left']
    north, east = bbox['top'], bbox['right']

    # Generate base map
    generator = OSMMapGenerator(color_palette=color_palette)
    print("Fetching map data...")
    data = generator.fetch_map_data(south, west, north, east)
    
    print("Processing features...")
    features = generator.process_data(data)
    
    description = generator.generate_map_description(features, "base")
    print("Map description:", description)

    prompt_to_llm = f'''
    Based on that info around an area of a map:
    {description}
    Write in one sentence what landscape is it. 
    Make it concise in one simple sentence, without any numbers or high details.
    Explain what landscape has and don't mention what it lacks. 
    '''
    # save prompt to file
    llm_output = run_openai_api(prompt_to_llm)
    print("LLM description:", llm_output)

    with open(f"{OUTPUT_DIR}/{timestamp}_llm.txt", "w") as f:
        f.write(f"Prompt:\n{prompt_to_llm}\n\nLLM Output:\n{llm_output}")

    # Create and save base map
    fig = generator.plot_map(features, south, west, north, east)
    fig.savefig(base_map_path, bbox_inches='tight', pad_inches=0)

    # Add building overlays
    overlay = BuildingOverlay()

    overlay.overlay_object_icons(
        base_map_path, 
        map_with_buildings_path, 
        features,  
        south, 
        west, 
        north, 
        east
    )
    
    base_prompt = """Create a detailed fantasy map for a Dungeons & Dragons (D&D) campaign. Pixel art style. The map should be depicted from a top-down perspective, allowing for a clear view of the terrain, topography, and landmarks rivers, valleys, or cities."""
    
    prompt = f"{base_prompt} {llm_output}"
    print("Using prompt:", prompt)

    init_image = load_image(map_with_buildings_path).resize((1024, 1024))
    generated_images = []

    for i in range(params['inference_count']):
        print(f"Generating variation {i+1}/{params['inference_count']}...")
        if USE_MLX:
            generated = model.generate_image(
                seed=random.randint(0, 1000000),
                prompt=prompt,
                config=Config(
                    num_inference_steps=params['inference_steps'],
                    height=1024,
                    width=1024,
                    init_image_path=map_with_buildings_path,
                    init_image_strength=params['strength'],
                    guidance=params['guidance_scale']
                )
            )   
            image = generated.image
        else:
            image = model(
                prompt=prompt, 
                image=init_image,
                num_inference_steps=params['inference_steps'], 
                strength=params['strength'], 
                guidance_scale=params['guidance_scale'],
                num_images_per_prompt=1,
            ).images[0]
        
        # Save generated image
        image_path = f"{OUTPUT_DIR}/{timestamp}_fantasy_{i+1}.png"
        image.save(image_path)
        generated_images.append(image_path)
        print(f"Saved generated image to {image_path}")

    return generated_images


def generate_random_bbox(bbox_limits):
    """
    Generate a random square bounding box within the given limits.
    The box size will be maximum 1/10 of the total area, maintaining square aspect ratio.
    """
    # Calculate total dimensions
    total_lat_span = bbox_limits['top'] - bbox_limits['bottom']
    total_lon_span = bbox_limits['right'] - bbox_limits['left']
    
    # Calculate the smaller dimension to ensure box fits within bounds
    max_size = min(total_lat_span, total_lon_span) / 10
    
    # Generate random box size (between 50% and 100% of max size)
    box_size = random.uniform(max_size * 0.5, max_size)
    
    # Generate random position (ensuring box stays within limits)
    bottom = random.uniform(
        bbox_limits['bottom'],
        bbox_limits['top'] - box_size
    )
    left = random.uniform(
        bbox_limits['left'],
        bbox_limits['right'] - box_size
    )
    
    random_bbox = {
        'top': bottom + box_size,
        'bottom': bottom,
        'left': left,
        'right': left + box_size
    }
    
    return random_bbox


def create_results_subplot(all_results):
    """
    Create a subplot with all generated images
    all_results: dict with structure {location_index: {palette_name: [image_paths]}}
    """
    num_locations = len(all_results)
    num_palettes = len(all_results[0])  # Number of color palettes
    num_variations = len(all_results[0][list(all_results[0].keys())[0]])  # Number of variations per palette
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 4 * num_locations))
    fig.suptitle('Generated Fantasy Maps by Location and Style', fontsize=16)
    
    for loc_idx in range(num_locations):
        for palette_idx, (palette_name, image_paths) in enumerate(all_results[loc_idx].items()):
            for var_idx, img_path in enumerate(image_paths):
                # Calculate subplot position
                plot_idx = loc_idx * (num_palettes * num_variations) + palette_idx * num_variations + var_idx + 1
                
                ax = fig.add_subplot(num_locations, num_palettes * num_variations, plot_idx)
                img = Image.open(img_path)
                ax.imshow(np.array(img))
                if var_idx == 0:
                    ax.set_title(f'Location {loc_idx + 1}\n{palette_name}')
                ax.axis('off')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = f"{OUTPUT_DIR}/{timestamp}_results_grid.png"
    plt.savefig(results_path, bbox_inches='tight', dpi=300)
    print(f"Results grid saved to: {results_path}")
    plt.close()

def main():
    # print(run_clarin_api("Hello how are you?"))

    bbox_limits = {
        'top': 51.294559,
        'bottom': 50.902167,
        'left': 16.642914,
        'right': 17.440796
    }

    params = {
        'strength': 0.65,
        'guidance_scale': 7.0,
        'inference_steps': 25,
        'inference_count': 3
    }

    # Load model
    model = load_model(device)

    # Generate maps with different color palettes and random boxes
    color_palettes = {
        "Forest Realm": MapColorPalettes.FOREST_REALM,
        "Winter's Feast": MapColorPalettes.OCEAN_KINGDOM,
        "Desert Empire": MapColorPalettes.DESERT_EMPIRE
    }

    num_locations = 50   # Number of random locations to generate
    all_results = {}     # Store all generated image paths
    
    for loc_idx in range(num_locations):
        print(f"\n\n\n\nGenerating maps for location {loc_idx + 1}/{num_locations}")
        random_bbox = generate_random_bbox(bbox_limits)
        print(f"Random box: {random_bbox}")
        
        with open(f"{OUTPUT_DIR}/random_bbox_{loc_idx}.txt", "w") as f:
            f.write(str(random_bbox))       # Saving location coordinates
        
        all_results[loc_idx] = {}
        
        for palette_name, palette in color_palettes.items():
            print(f"\nTesting with {palette_name} palette...")
            generated_paths = generate_fantasy_map(random_bbox, palette, model, params)
            all_results[loc_idx][palette_name] = generated_paths
            print(f"Generated {len(generated_paths)} images for {palette_name}")

    create_results_subplot(all_results)

if __name__ == "__main__":
    main()