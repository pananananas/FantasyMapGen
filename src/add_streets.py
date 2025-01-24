from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests

class StreetNameOverlay:
    def __init__(self, overpass_url="http://overpass-api.de/api/interpreter"):
        self.overpass_url = overpass_url

    def fetch_street_data(self, south, west, north, east):
        query = f"""
        [out:json][timeout:25];
        (
            way["highway"]["name"]({south},{west},{north},{east});
        );
        out body;
        >;
        out skel qt;
        """
        response = requests.get(self.overpass_url, params={'data': query})
        return response.json()

    def extract_street_names(self, data):
        nodes = {node['id']: (node['lon'], node['lat']) for node in data['elements'] if node['type'] == 'node'}
        streets = []
        for element in data['elements']:
            if element['type'] == 'way' and 'tags' in element and 'name' in element['tags']:
                street_name = element['tags']['name']
                coords = [nodes[node_id] for node_id in element['nodes'] if node_id in nodes]
                if coords:
                    streets.append((street_name, coords))
        return streets

    def overlay_street_names(self, map_image_path, output_image_path, streets, font_path=None):
        # Load the map image
        map_image = Image.open(map_image_path)
        draw = ImageDraw.Draw(map_image)

        # Set font
        if font_path:
            font = ImageFont.truetype(font_path, size=12)
        else:
            font = ImageFont.load_default()

        # Map bounds
        min_lat, max_lat = 51.1050, 51.1150
        min_lon, max_lon = 17.0520, 17.0670

        # Track unique street names
        written_street_names = set()

        # Overlay street names
        for name, coords in streets:
            if name not in written_street_names:
                # Convert geographical coordinates to pixel coordinates
                pixel_coords = [
                    (
                        (lon - min_lon) / (max_lon - min_lon) * map_image.width,
                        (1 - (lat - min_lat) / (max_lat - min_lat)) * map_image.height
                    )
                    for lon, lat in coords
                ]

                # Determine the position for the street name (middle of the segment)
                if len(pixel_coords) > 1:
                    mid_idx = len(pixel_coords) // 2
                    text_position = pixel_coords[mid_idx]
                    draw.text(text_position, name, fill="black", font=font)

                    # Mark the street name as written
                    written_street_names.add(name)

        # Save the output image
        map_image.save(output_image_path)

if __name__ == "__main__":
    # Define bounding box for the region
    south, west = 51.1050, 17.0520
    north, east = 51.1150, 17.0670

    # Path to the existing map image and output image
    map_image_path = "data/maps/gen_map.png"
    output_image_path = "data/maps/overlay_with_street_names.png"

    # Fetch and overlay street names
    overlay = StreetNameOverlay()
    print("Fetching street data...")
    street_data = overlay.fetch_street_data(south, west, north, east)

    print("Extracting street names...")
    streets = overlay.extract_street_names(street_data)

    print("Overlaying street names...")
    font_path = "C:\\WINDOWS\\FONTS\\ARIAL.ttf"
    overlay.overlay_street_names(map_image_path, output_image_path, streets, font_path=font_path)

    print(f"Overlay complete. Saved to {output_image_path}.")

