from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import requests
import os

class MapColorPalettes:
    DEFAULT = {
        'water': ('blue', 0.3),
        'green': ('lightgreen', 0.3),   
        'forest': ('darkgreen', 0.3),
        'farmland': ('yellowgreen', 0.2),
        'residential': ('lightgray', 0.2),
        'commercial': ('orange', 0.2),
        'industrial': ('brown', 0.2),
        'buildings': ('red', 0.3),
        'leisure': ('pink', 0.3),
        'main_roads': ('gray', 1.0),
        'secondary_roads': ('lightgray', 1.0),
        'pedestrian': ('beige', 0.5),
        'railway': ('black', 1.0),
        'amenities': ('purple', 0.5),
        'background': 'tan'
    }

    DARK = {
        'water': ('#b71c1c', 0.4),  # Dark red
        'green': ('#1b5e20', 0.4),  # Dark green
        'forest': ('#004d40', 0.4),  # Darker green
        'farmland': ('#33691e', 0.3),  # Olive green
        'residential': ('#424242', 0.3),  # Dark gray
        'commercial': ('#e65100', 0.3),  # Dark orange
        'industrial': ('#3e2723', 0.3),  # Dark brown
        'buildings': ('#101010', 0.4),  # Dark gray
        'leisure': ('#880e4f', 0.4),  # Dark pink
        'main_roads': ('#fafafa', 1.0),  # Light gray
        'secondary_roads': ('#C28A4E', 1.0),  # Orange
        'pedestrian': ('#bdbdbd', 0.5),  # Medium gray
        'railway': ('#C28A4E', 1.0),  # Orange
        'amenities': ('#ce93d8', 0.6),  # Light purple
        'background': '#263238'  # Dark blue-gray
    }

    VINTAGE = {
        'water': ('#a8ccd7', 0.4),  # Pale blue
        'green': ('#98a886', 0.4),  # Sage green
        'forest': ('#6b7f5e', 0.4),  # Olive
        'farmland': ('#c5b7a0', 0.3),  # Beige
        'residential': ('#d9d2c6', 0.3),  # Light beige
        'commercial': ('#c17f59', 0.3),  # Terra cotta
        'industrial': ('#8e7761', 0.3),  # Brown
        'buildings': ('#ab6c6c', 0.4),  # Dusty red
        'leisure': ('#d4a5a5', 0.4),  # Rose
        'main_roads': ('#847c72', 1.0),  # Warm gray
        'secondary_roads': ('#a39d93', 1.0),  # Light warm gray
        'pedestrian': ('#c2bab0', 0.5),  # Very light warm gray
        'railway': ('#585652', 1.0),  # Dark warm gray
        'amenities': ('#9c849c', 0.6),  # Muted purple
        'background': '#f5e6d3'  # Cream
    }

    PASTEL = {
        'water': ('#bbdefb', 0.4),  # Light blue
        'green': ('#c8e6c9', 0.4),  # Light green
        'forest': ('#a5d6a7', 0.4),  # Mint
        'farmland': ('#dcedc8', 0.3),  # Light lime
        'residential': ('#f5f5f5', 0.3),  # White smoke
        'commercial': ('#ffe0b2', 0.3),  # Light orange
        'industrial': ('#d7ccc8', 0.3),  # Light brown
        'buildings': ('#ffcdd2', 0.4),  # Light red
        'leisure': ('#f8bbd0', 0.4),  # Light pink
        'main_roads': ('#9e9e9e', 1.0),  # Medium gray
        'secondary_roads': ('#bdbdbd', 1.0),  # Light gray
        'pedestrian': ('#e0e0e0', 0.5),  # Very light gray
        'railway': ('#757575', 1.0),  # Dark gray
        'amenities': ('#e1bee7', 0.6),  # Light purple
        'background': '#fff3e0'  # Light orange
    }

    FOREST_REALM = {
        'water': ('#4a777a', 0.6),  # Muted teal
        'green': ('#2d5a27', 0.5),  # Deep forest green
        'forest': ('#1e3f1f', 0.6),  # Dark forest green
        'farmland': ('#687d3e', 0.4),  # Moss green
        'residential': ('#a4b494', 0.4),  # Sage
        'commercial': ('#7d8471', 0.4),  # Green grey
        'industrial': ('#4a5d4c', 0.4),  # Dark sage
        'buildings': ('#2d3a2d', 0.5),  # Forest shadow
        'leisure': ('#789268', 0.4),  # Light forest green
        'main_roads': ('#d8bc83', 0.8),  # Light dirt path
        'secondary_roads': ('#b89b6a', 0.7),  # Dirt path
        'pedestrian': ('#a18e6e', 0.5),  # Faint trail
        'railway': ('#47402e', 0.9),  # Dark wood
        'amenities': ('#2d5a27', 0.7),  # Forest green markers
        'background': '#ccd3c6'  # Pale sage background
    }

    OCEAN_KINGDOM = {
        'water': ('#1e4875', 0.6),  # Deep ocean blue
        'green': ('#5b8b95', 0.4),  # Sea sage
        'forest': ('#2f6d7e', 0.5),  # Ocean forest
        'farmland': ('#7ab0b7', 0.4),  # Coastal fields
        'residential': ('#a4c3d2', 0.4),  # Coastal town
        'commercial': ('#6494aa', 0.4),  # Harbor blue
        'industrial': ('#4d7285', 0.4),  # Port blue
        'buildings': ('#2c4a5a', 0.5),  # Deep marine
        'leisure': ('#8fb8c9', 0.4),  # Light sea blue
        'main_roads': ('#d5d9db', 0.8),  # Pearl white
        'secondary_roads': ('#b8c4c9', 0.7),  # Shell white
        'pedestrian': ('#9ba7ad', 0.5),  # Misty path
        'railway': ('#2d4047', 0.9),  # Dark marine
        'amenities': ('#1e4875', 0.7),  # Ocean blue markers
        'background': '#e6eef2'  # Pale ocean mist
    }

    DESERT_EMPIRE = {
        'water': ('#4d9078', 0.5),  # Oasis green
        'green': ('#b8a364', 0.4),  # Desert grass
        'forest': ('#8b7355', 0.5),  # Desert woods
        'farmland': ('#d4ba7d', 0.4),  # Sandy fields
        'residential': ('#e6d5ac', 0.4),  # Sand town
        'commercial': ('#c4a775', 0.4),  # Market sand
        'industrial': ('#a68a5b', 0.4),  # Work yards
        'buildings': ('#8c7356', 0.5),  # Clay buildings
        'leisure': ('#d6c291', 0.4),  # Light sand
        'main_roads': ('#f2e4c1', 0.8),  # Main caravan route
        'secondary_roads': ('#e6d5ac', 0.7),  # Desert path
        'pedestrian': ('#d4c398', 0.5),  # Sand trail
        'railway': ('#6b5642', 0.9),  # Dark desert
        'amenities': ('#b8a364', 0.7),  # Desert markers
        'background': '#f5ecd6'  # Pale sand
    }

class OSMMapGenerator:
    def __init__(self, color_palette=MapColorPalettes.DEFAULT):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.color_palette = color_palette

    def fetch_map_data(self, south, west, north, east):
        query = f"""
        [out:json][timeout:25];
        (
            // Water features
            way["natural"="water"]({south},{west},{north},{east});
            relation["natural"="water"]({south},{west},{north},{east});
            
            // Green and natural areas
            way["leisure"="park"]({south},{west},{north},{east});
            way["landuse"="grass"]({south},{west},{north},{east});
            way["landuse"="forest"]({south},{west},{north},{east});
            way["landuse"="farmland"]({south},{west},{north},{east});
            way["natural"="scrub"]({south},{west},{north},{east});
            way["natural"="grassland"]({south},{west},{north},{east});
            
            // All roads
            way["highway"]({south},{west},{north},{east});
            
            // Buildings and structures
            way["building"]({south},{west},{north},{east});
            
            // Points of interest
            node["amenity"]({south},{west},{north},{east});
            
            // Railway
            way["railway"]({south},{west},{north},{east});
            
            // Leisure facilities
            way["leisure"]({south},{west},{north},{east});
            
            // Commercial areas
            way["landuse"="commercial"]({south},{west},{north},{east});
            
            // Industrial areas
            way["landuse"="industrial"]({south},{west},{north},{east});
            
            // Residential areas
            way["landuse"="residential"]({south},{west},{north},{east});
        );
        out body;
        >;
        out skel qt;
        """
        
        response = requests.get(self.overpass_url, params={'data': query})
        return response.json()

    def process_data(self, data):
        nodes = {node['id']: (node['lon'], node['lat']) 
                for node in data['elements'] if node['type'] == 'node'}
        
        features = {
            'water': [],
            'green': [],
            'forest': [],
            'farmland': [],
            'residential': [],
            'commercial': [],
            'industrial': [],
            'main_roads': [],
            'secondary_roads': [],
            'pedestrian': [],
            'buildings': [],
            'railway': [],
            'leisure': [],
            'amenities': [],
            'hospitals': [],
            'schools': [],
            'shops': []
        }
        
        for element in data['elements']:
            if element['type'] == 'way' and 'nodes' in element:
                coords = [nodes[node_id] for node_id in element['nodes']]
                if len(coords) < 2:
                    continue
                
                if 'tags' in element:
                    tags = element['tags']
                    
                    # Process various types of features
                    if tags.get('natural') == 'water':
                        features['water'].append(coords)
                    
                    elif tags.get('landuse') in ['grass', 'forest']:
                        features['forest'].append(coords)
                    elif tags.get('leisure') == 'park':
                        features['green'].append(coords)
                    elif tags.get('landuse') == 'farmland':
                        features['farmland'].append(coords)
                    
                    elif 'highway' in tags:
                        if tags['highway'] in ['motorway', 'trunk', 'primary']:
                            features['main_roads'].append(coords)
                        elif tags['highway'] in ['secondary', 'tertiary']:
                            features['secondary_roads'].append(coords)
                        elif tags['highway'] in ['pedestrian', 'footway', 'path']:
                            features['pedestrian'].append(coords)
                    
                    elif 'building' in tags:
                        features['buildings'].append(coords)
                    
                    elif 'railway' in tags:
                        features['railway'].append(coords)
                    
                    elif tags.get('landuse') == 'residential':
                        features['residential'].append(coords)
                    elif tags.get('landuse') == 'commercial':
                        features['commercial'].append(coords)
                    elif tags.get('landuse') == 'industrial':
                        features['industrial'].append(coords)
                    
                    elif 'leisure' in tags:
                        features['leisure'].append(coords)

                    # Process hospitals, schools, and shops
                    if 'amenity' in tags:
                        if tags['amenity'] == 'hospital':
                            features['hospitals'].append(coords)
                        elif tags['amenity'] == 'school':
                            features['schools'].append(coords)
                        elif tags['amenity'] == 'shop':
                            features['shops'].append(coords)
                
            elif element['type'] == 'node' and 'tags' in element and 'amenity' in element['tags']:
                tags = element['tags']
                if tags.get('amenity') == 'hospital':
                    features['hospitals'].append((element['lon'], element['lat']))
                elif tags.get('amenity') == 'school':
                    features['schools'].append((element['lon'], element['lat']))
                elif tags.get('amenity') == 'shop':
                    features['shops'].append((element['lon'], element['lat']))
        
        return features

    def plot_map(self, features, south, west, north, east):
        """Create a detailed visualization of the map features"""
        dpi = 300
        fig = plt.figure(figsize=(8, 8), dpi=dpi, facecolor=self.color_palette['background'])
        ax = fig.add_subplot(111)

        # Plot polygonal features
        for feature_type in ['water', 'green', 'forest', 'farmland', 
                           'commercial', 'industrial', 'leisure']:
            if feature_type in features and features[feature_type]:
                color, alpha = self.color_palette[feature_type]
                for coords in features[feature_type]:
                    if len(coords) >= 3:
                        poly = Polygon(coords, facecolor=color, alpha=alpha, edgecolor='none')
                        ax.add_patch(poly)
        
        # Plot linear features
        if features['main_roads']:
            color, alpha = self.color_palette['main_roads']
            for coords in features['main_roads']:
                coords_array = np.array(coords)
                ax.plot(coords_array[:, 0], coords_array[:, 1], color=color, linewidth=6)
        
        if features['secondary_roads']:
            color, alpha = self.color_palette['secondary_roads']
            for coords in features['secondary_roads']:
                coords_array = np.array(coords)
                ax.plot(coords_array[:, 0], coords_array[:, 1], color=color, linewidth=2)
        
        # if features['pedestrian']:
        #     color, alpha = self.color_palette['pedestrian']
        #     for coords in features['pedestrian']:
        #         coords_array = np.array(coords)
                # ax.plot(coords_array[:, 0], coords_array[:, 1], color=color, linewidth=0.5, linestyle='--')
        
        # if features['railway']:
        #     color, alpha = self.color_palette['railway']
        #     for coords in features['railway']:
        #         coords_array = np.array(coords)
                # ax.plot(coords_array[:, 0], coords_array[:, 1], color=color, linewidth=1, linestyle='--')
        
        # if features['amenities']:
        #     color, alpha = self.color_palette['amenities']
        #     amenity_coords = np.array(features['amenities'])
            # ax.scatter(amenity_coords[:, 0], amenity_coords[:, 1], c=color, s=10, alpha=alpha)

        min_lon = west
        max_lon = east
        min_lat = south
        max_lat = north
        
        # Set exact bounds from the coordinates you provided
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        
        # Ensure aspect ratio is equal (square plot)
        ax.set_aspect('equal')
        plt.axis('off')
        
        # Remove padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        return fig
    

    def generate_map_description(self, processed_data, map_name):
        """
        Tworzy opis mapy na podstawie przetworzonych danych.
        """
        description = []

        # Woda
        water_count = len(processed_data['water'])
        if water_count > 0:
            description.append(f"There are {water_count} bodies of water")

        # Budynki
        buildings_count = len(processed_data['buildings'])
        if buildings_count > 0:
            description.append(f"The map contains {buildings_count} buildings")

        # Drogi
        main_roads_count = len(processed_data['main_roads'])
        secondary_roads_count = len(processed_data['secondary_roads'])
        if main_roads_count > 0 or secondary_roads_count > 0:
            description.append(
                f"There are {main_roads_count} main roads and {secondary_roads_count} secondary roads"
            )

        # Las i zieleń
        forest_count = len(processed_data['forest'])
        if forest_count > 0:
            description.append(f"The area includes {forest_count} forest regions")

        
        # Parki i obiekty rekreacyjne
        leisure_count = len(processed_data['leisure'])
        if leisure_count > 0:
            description.append(f"The map contains {leisure_count} leisure areas such as parks")

        # Obiekty komercyjne
        commercial_count = len(processed_data['commercial'])
        if commercial_count > 0:
            description.append(f"The map includes {commercial_count} commercial areas")

        # Obiekty przemysłowe
        industrial_count = len(processed_data['industrial'])
        if industrial_count > 0:
            description.append(f"There are {industrial_count} industrial zones")

        # Obszary mieszkaniowe
        residential_count = len(processed_data['residential'])
        if residential_count > 0:
            description.append(f"The map has {residential_count} residential areas")

        # Tereny rolne
        farmland_count = len(processed_data['farmland'])
        if farmland_count > 0:
            description.append(f"The area includes {farmland_count} farmland regions")

        # Szpitale
        hospitals_count = len(processed_data['hospitals'])
        if hospitals_count > 0:
            description.append(f"There are {hospitals_count} hospitals")

        # Szkoły
        schools_count = len(processed_data['schools'])
        if schools_count > 0:
            description.append(f"The map contains {schools_count} schools")

        # Sklepy
        shops_count = len(processed_data['shops'])
        if shops_count > 0:
            description.append(f"There are {shops_count} shops")

        return description
    

class BuildingOverlay:
    def __init__(self, icons_folder='assets/icons'):
        self.icons_folder = icons_folder

    def extract_building_coordinates(self, buildings):
        """
            Extracts the center coordinates of buildings from the provided building polygons.
        """
        building_centers = []
        for coords in buildings:
            # Logowanie, żeby zobaczyć dane wejściowe
            #print(f"Processing feature coordinates: {coords}")

            if isinstance(coords, tuple) and len(coords) == 2:
                # Jeśli to pojedynczy punkt, po prostu dodajemy go
                building_centers.append(coords)
            elif isinstance(coords, list):
                # Jeśli to lista współrzędnych, obliczamy środek
                center_lon = sum(coord[0] for coord in coords) / len(coords)
                center_lat = sum(coord[1] for coord in coords) / len(coords)
                building_centers.append((center_lon, center_lat))
            else:
                print(f"Invalid data format: {coords}")
        return building_centers

    def cluster_buildings(self, buildings, threshold):
        """
        Clusters buildings based on proximity.
        """
        clusters = []
        used = set()

        for i, (lon1, lat1) in enumerate(buildings):
            if i in used:
                continue

            cluster = [(lon1, lat1)]
            used.add(i)

            for j, (lon2, lat2) in enumerate(buildings):
                if j not in used:
                    distance = np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
                    if distance < threshold:
                        cluster.append((lon2, lat2))
                        used.add(j)

            clusters.append(cluster)

        return clusters

    def overlay_object_icons(self, map_image_path, output_image_path, features, south, west, north, east, icon_scale=1.2, cluster_scale_factor=0.005, threshold=0.0015):
        """
        Overlays object icons onto a map image based on their coordinates and object type.
        """
        # Load the map image
        map_image = Image.open(map_image_path)
        draw = ImageDraw.Draw(map_image)

        # Map bounds
        min_lat, max_lat = south, north
        min_lon, max_lon = west, east

        # Process each feature in the features dictionary
        for feature_type, feature_data in features.items():
            if feature_type == 'water': continue  # Skip water for this implementation

            # Determine icon path
            icon_filename = f'{feature_type}.png' if feature_type not in ['main_roads', 'secondary_roads'] else 'road.png'
            icon_path = f'{self.icons_folder}/{icon_filename}'

            # Load the icon
            try:
                icon = Image.open(icon_path)
            except FileNotFoundError:
                #print(f"Icon for {feature_type} not found. Skipping.")
                continue

            # Extract feature centers for the current feature
            feature_centers = self.extract_building_coordinates(feature_data)

            # Cluster buildings for this feature
            clusters = self.cluster_buildings(feature_centers, threshold)

            for cluster in clusters:
                # Calculate the cluster center
                center_lon = sum(lon for lon, lat in cluster) / len(cluster)
                center_lat = sum(lat for lon, lat in cluster) / len(cluster)

                # Scale the icon size based on the cluster size
                scale_factor = icon_scale + cluster_scale_factor * (len(cluster) - 1)
                resized_icon_width = int(icon.width * scale_factor)
                resized_icon_height = int(icon.height * scale_factor)
                resized_icon = icon.resize((resized_icon_width, resized_icon_height), Image.Resampling.LANCZOS)

                # Convert geographical coordinates to pixel coordinates
                x = (center_lon - min_lon) / (max_lon - min_lon) * map_image.width
                y = (1 - (center_lat - min_lat) / (max_lat - min_lat)) * map_image.height

                # Calculate position to paste the icon
                top_left_x = int(x - resized_icon_width / 2)
                top_left_y = int(y - resized_icon_height / 2)

                # Paste the resized icon onto the map
                map_image.paste(resized_icon, (top_left_x, top_left_y), resized_icon.convert("RGBA"))

        # Save the output image
        map_image.save(output_image_path)
        return map_image


if __name__ == "__main__":
    
    # Wrocław centrum
    south, west = 51.1050, 17.0520
    north, east = 51.1150, 17.0670

    generator = OSMMapGenerator(color_palette=MapColorPalettes.DARK)

    print("Fetching map data...")
    data = generator.fetch_map_data(south, west, north, east)

    print("Processing features...")
    features = generator.process_data(data)

    print("Creating visualization...")
    fig = generator.plot_map(features, south, west, north, east)

    map_name = 'map_dark'

    print("...")
    description = generator.generate_map_description(features, map_name)
    print(description)

    print("...")
    # llm_description = generator.analyze_with_llm(description)
    # print(llm_description)

    fig.savefig(f'{map_name}.png', bbox_inches='tight', pad_inches=0)

    # Ścieżka do folderu 'icons' (jako folder nadrzędny względem 'src')
    icons_folder = os.path.join(os.path.dirname(__file__), '..', 'assets/icons')


    print("Overlaying object icons...")
    overlay = BuildingOverlay(icons_folder)  # Określenie folderu z ikonami

    # Nałożenie ikon dla wszystkich obiektów
    overlay.overlay_object_icons(f'{map_name}.png', f'{map_name}_with_objects.png', features, south, west, north, east)