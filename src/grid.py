from PIL import Image, ImageDraw

def draw_dnd_grid(input_image_path, output_image_path, grid_size, line_color=(0, 0, 0, 255), line_width=2):
    """
    Overlays a D&D combat grid onto a map image.

    :param input_image_path: Path to the input map image
    :param output_image_path: Path to save the output image with grid
    :param grid_size: Size of each grid cell (in pixels)
    :param line_color: Color of the grid lines (RGBA tuple)
    :param line_width: Width of the grid lines
    """
    image = Image.open(input_image_path).convert("RGBA")
    width, height = image.size

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    combined = Image.alpha_composite(image, overlay)

    combined.convert("RGB").save(output_image_path, "PNG")
    print(f"Grid added and saved to {output_image_path}")


# Example usage
if __name__ == "__main__":
    input_map = "data/maps/map.png"
    output_map = "map_with_grid.png"

    # Grid settings
    grid_cell_size = 50
    grid_line_color = (0, 0, 0, 255)
    grid_line_width = 2

    draw_dnd_grid(input_map, output_map, grid_cell_size, grid_line_color, grid_line_width)