import random
from src.draw_map import BuildingOverlay, MapColorPalettes, OSMMapGenerator
from diffusers import FluxImg2ImgPipeline
from streamlit_folium import st_folium
from diffusers.utils import load_image
from mflux import Flux1, Config
from folium.plugins import Draw
from datetime import datetime
from openai import OpenAI
import streamlit as st
import requests
import folium
import torch
import os

global pipe
global pipe_cuda
global model_config

USE_MLX = True

print("imports done")

# Initialize session state variables if they don't exist
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

if 'bbox' not in st.session_state:
    st.session_state['bbox'] = None

if not st.session_state['initialized']:
    os.makedirs("data/streamlit", exist_ok=True)
    st.session_state['datetime'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state['map_path'] = f"data/streamlit/{st.session_state['datetime']}_map.png"
    st.session_state['map_path_with_buildings'] = f"data/streamlit/{st.session_state['datetime']}_map_with_buildings.png"
    st.session_state['generated_map_path'] = f"data/streamlit/{st.session_state['datetime']}_fantasy.png"
    st.session_state['device'] = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    st.session_state['initialized'] = True

# Use session state variables instead of global variables
map_path = st.session_state['map_path']
map_path_with_buildings = st.session_state['map_path_with_buildings']
generated_map_path = st.session_state['generated_map_path']
device = st.session_state['device']

def run_clarin_api(prompt):

    url = "https://services.clarin-pl.eu/api/v1/oapi/chat/completions"
    user_token = os.getenv('CLARIN_API_KEY')
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    data = response.json()
    
    return data['choices'][0]['message']['content']



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




st.set_page_config(
    page_title="Interactive Map Selector",
    page_icon="🗺️",
    layout="wide"
)

st.title("Interactive Map Area Selector 🗺️ -> 🧝‍♀️")

st.markdown("""
### How to use:
1. Use the rectangle tool from the drawing toolbar on the top left
2. Click and drag on the map to draw a rectangle
3. Select style of your map
4. Click the 'Select area' button to see the coordinates
5. Wait for the map to be generated
6. Download the map by clicking the 'Download' button
""") 

# Create a map centered on a default location (Wrocław)
m = folium.Map(location=[51.1079, 17.0385], zoom_start=13)

# Add the draw control to the map - only rectangle allowed
draw = Draw(
    draw_options={
        'polyline': False,
        'circle': False,
        'circlemarker': False,
        'marker': False,
        'polygon': False,
        'rectangle': True,
    },
    edit_options={'edit': False}
)
draw.add_to(m)

map_data = st_folium(m, width=1200, height=600)


color_palettes = {
    "Forest Realm": MapColorPalettes.FOREST_REALM,
    "Winter's Feast": MapColorPalettes.OCEAN_KINGDOM,
    "Desert Empire": MapColorPalettes.DESERT_EMPIRE,
    "Vintage": MapColorPalettes.VINTAGE,
    "Default": MapColorPalettes.DEFAULT,
    "Pastel": MapColorPalettes.PASTEL,
    "Dark": MapColorPalettes.DARK,
}

color_pallete = st.selectbox("Select color palette", list(color_palettes.keys()))


col1, col2, col3, col4 = st.columns(4)
with col1:
    strength = st.slider("Strength", 0.0, 1.0, 0.65)
with col2:
    guidance_scale = st.slider("Guidance scale", 1.0, 10.0, 7.0)
with col3:
    inference_steps = st.slider("Inference steps", 1, 30, 20)
with col4:
    inference_count = st.slider("Inference count", 1, 10, 4)

base_prompt = """Create a detailed fantasy map for a Dungeons & Dragons (D&D) campaign. The map should be depicted from a top-down perspective, allowing for a clear view of the terrain, topography, and landmarks rivers, valleys, or cities."""


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Button for selecting area
if st.button("Select area"):
    if map_data['last_active_drawing']:
        if map_data['last_active_drawing']['geometry']['type'] == 'Polygon':
            coordinates = map_data['last_active_drawing']['geometry']['coordinates'][0]
            
            # Extract all latitudes and longitudes
            lats = [coord[1] for coord in coordinates]
            lons = [coord[0] for coord in coordinates]
            
            # Calculate bounding box
            top = max(lats)
            bottom = min(lats)
            left = min(lons)
            right = max(lons)
            print(f"Bounding box: top={top}, bottom={bottom}, left={left}, right={right}")
            
            # Store these values in session state
            st.session_state['bbox'] = {
                'top': top,
                'bottom': bottom,
                'left': left,
                'right': right
            }
    else:
        st.warning("Please draw a rectangle on the map first!")
if 'bbox' not in st.session_state:
    st.session_state['bbox'] = None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Loading the model 
@st.cache_resource
def load_model(device):
    """Load the model once and cache it"""
    print("Loading model...")
    if USE_MLX:
        model_config = Flux1.from_alias("dev").model_config
        pipe = Flux1(
            model_config=model_config,
            quantize=4,
            lora_paths=["loras/lora_dev_10000.safetensors"],
            lora_scales=[1.0],
        )
        return pipe
    
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

# Move the model loading logic before the Streamlit UI elements
if 'flux_model' not in st.session_state:
    with st.status("Initializing AI model...", expanded=False) as status:
        st.write("Loading model configuration...")
        
        # Load model using cached function
        st.session_state['flux_model'] = load_model(device)
        
        status.update(label=f"AI model initialized! on device: {device}", state="complete")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Map generation
if st.session_state['bbox'] is not None:
    south, west = st.session_state['bbox']['bottom'], st.session_state['bbox']['left']
    north, east = st.session_state['bbox']['top'], st.session_state['bbox']['right']

    # Initialize session state variables if they don't exist
    if 'base_map_generated' not in st.session_state:
        st.session_state['base_map_generated'] = False
        st.session_state['llm_description'] = None

    generator = OSMMapGenerator(color_palette=color_palettes[color_pallete])

    if not st.session_state['base_map_generated']:
        with st.status("Creating base map...", expanded=False) as status:
            st.write("Fetching map data...")
            data = generator.fetch_map_data(south, west, north, east)
            status.update(label="Map data fetched!", state="complete")

            st.write("Processing features...")
            features = generator.process_data(data)
            status.update(label="Features processed!", state="complete")

            description = generator.generate_map_description(features, "base")
            print("Description of the map:")
            print(description)

            prompt_to_llm = f'''
Based on that info around an area of a map:
{description}
Write in one sentence what landscape is it. 
Make it concise in one simple sentence, without any numbers or high details.
Explain what landscape has and don't mention what it lacks. 
'''
            llm_output = run_openai_api(prompt_to_llm)
            # llm_output = run_clarin_api(prompt_to_llm)
            st.session_state['llm_description'] = llm_output
            status.update(label="Landscape description generated!", state="complete")

            st.write("Creating visualization...")
            fig = generator.plot_map(features, south, west, north, east)
            status.update(label="Visualization created!", state="complete")

            fig.savefig(map_path, bbox_inches='tight', pad_inches=0)
            status.update(label="Base map saved!", state="complete")

            overlay = BuildingOverlay()
            building_centers = overlay.extract_building_coordinates(features['buildings'])
            status.update(label="Building coordinates extracted!", state="complete")

            print("Overlaying building icons...")
            # overlay_object_icons(self, map_image_path, output_image_path, features, south, west, north, east, icon_scale=1.2, cluster_scale_factor=0.005, threshold=0.0015):
            overlay.overlay_object_icons(
                map_path, 
                map_path_with_buildings, 
                features,  
                south, 
                west, 
                north, 
                east
            )
            status.update(label="Building icons overlaid!", state="complete")

            st.session_state['base_map_generated'] = True
            status.update(label="Base map created!", state="complete")

    st.image(map_path_with_buildings, caption='Base map created with OSM data')

    prompt = f"{base_prompt} {st.session_state['llm_description']}"
    
    st.text_area("Prompt based on map description", value=prompt)

    # Add Generate button
    if st.button("Generate Fantasy Map"):
        with st.status("Generating fantasy map...", expanded=False) as status:
            
            with st.spinner("Generating fantasy map... This may take a few minutes."):
                init_image = load_image(map_path_with_buildings).resize((1024, 1024))

                # Create a container for displaying images
                image_container = st.container()
                
                # Generate multiple images based on inference_count
                generated_images = []
                
                for i in range(inference_count):
                    if USE_MLX:
                        generated = st.session_state['flux_model'].generate_image(
                            seed=random.randint(0, 1000000),
                            prompt=prompt,
                            config=Config(
                                num_inference_steps=inference_steps,
                                height=1024,
                                width=1024,
                                init_image_path=map_path_with_buildings,
                                init_image_strength=strength,
                                guidance=guidance_scale
                            )
                        )   
                        image = generated.image
                    else:
                        image = st.session_state['flux_model'](
                            prompt=prompt, 
                            image=init_image,
                            num_inference_steps=inference_steps, 
                            strength=strength, 
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=1,
                        ).images[0]
                    
                    generated_images.append(image)
                    
                    # Display the image as soon as it's generated
                    with image_container:
                        st.image(image, caption=f'Fantasy map variation {i+1}')
                        # Save each generated image with a unique name
                        image_path = f"data/streamlit/{st.session_state['datetime']}_fantasy_{i+1}.png"
                        image.save(image_path)
                        
                        # Add download button for each image
                        st.download_button(
                            label=f"Download map variation {i+1}",
                            data=open(image_path, "rb"),
                            file_name=f"{st.session_state['datetime']}_fantasy_{i+1}.png",
                            mime="image/png"
                        )

                status.update(label=f"Generated {inference_count} fantasy maps!", state="complete")

else:
    st.info("Please select an area on the map and click 'Select area' to generate the map.")

st.markdown("---")

if st.button("Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    if os.path.exists(st.session_state.get('map_path', '')):
        os.remove(st.session_state['map_path'])
    if os.path.exists(st.session_state.get('generated_map_path', '')):
        os.remove(st.session_state['generated_map_path'])
    if os.path.exists(st.session_state.get('map_path_with_buildings', '')):
        os.remove(st.session_state['map_path_with_buildings'])
    
    # Force a rerun to reset all widgets
    st.rerun()

# Add a small note about what reset does
st.caption("Reset will clear all selections and generated images.")