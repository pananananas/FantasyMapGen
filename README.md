# MapGen - Fantasy Map Generator 🗺️ -> 🧝‍♀️

A sophisticated map generation tool that transforms real-world OpenStreetMap data into fantasy-style maps using AI image generation. Perfect for Dungeons & Dragons campaigns and fantasy world-building.

![MapGenDemo](https://github.com/user-attachments/assets/d96e97ad-4ade-45e2-86ae-335676db1f0b)

## Features

- 🌍 Interactive map selection through a Streamlit web interface
- 🎨 Multiple pre-defined color palettes:

  - 🌲 Forest Realm
  
  - ❄️ Winter's Feast

  - 🏜️ Desert Empire
      
  - 🏰 Vintage

  - 🌸 Pastel
  
  - 🔥 Hell

- 🏰 Automatic building and landmark detection
- 🎯 Customizable AI generation parameters
- 🔄 Multiple map variations per generation
- 📝 Intelligent map description generation
- 🎭 Fine-tuned AI model for fantasy-style map generation


# How it works?

1. **Map Selection**: User selects an area using an interactive map interface.
2. **Data Processing**: 
   - Fetches OpenStreetMap data for the selected region
   - Processes various features (buildings, roads, water bodies, etc.)
   - Generates a base map with the selected color style
   - Based on features of selected region, it generates a description using OpenAI API prompt for the AI image generation model
3. **AI Enhancement**:
   - Uses a fine-tuned FLUX model to transform the base map into fantasy style
   - Generates multiple variations with customizable parameters
   - Overlays buildings and landmarks with themed icons


# Running the project

## Installation

```bash
git clone https://github.com/pananananas/FantasyMapGen.git
cd FantasyMapGen
uv sync
```

The project is configured using uv package manager. [Here is a manual for installation](https://docs.astral.sh/uv/getting-started/installation/).

## Streamlit app

```bash
uv run streamlit_app.py
```

## Development Insights

- Exploring various diffusion models
- Fine-tuning the model on Reddit data
- Testing different styles of generated maps
- Evaluating the generation of descriptions of real maps to enhance terrain guidance for the diffusion model.

- Utilizing API data to place icons (e.g., tree, building, etc.) accurately on the map.
- Incorporating object information from the map into the prompt.

- Establishing the creative goal (the desired style for the system's output)
- Streamlit demonstration
- Enhancing the maps' utility for Dungeons & Dragons (DnD) gameplay.

## Fine-tuning Process:

Subreddits utilized for fine-tuning:

- r/dndmaps
- r/mapmaking
- r/DnDMapsSubmissions
- r/FantasyMaps
- r/imaginarymaps

Fine-tuning was conducted using [SimpleTuner](https://github.com/bghira/SimpleTuner/tree/main). 
Configuration files are available in `fine_tune_config` directory.
