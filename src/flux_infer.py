from mflux import Flux1, Config
import matplotlib.pyplot as plt
import random
import os

image_path = "data/maps/map.png"

# First get the model config
model_config = Flux1.from_alias("dev").model_config

# Then create Flux1 with the config and LoRA parameters
flux = Flux1(
    model_config=model_config,
    quantize=8,
    lora_paths=["loras/lora_dev_10000.safetensors"],
    lora_scales=[1.0],
)
prompt = "Create a detailed fantasy map for a Dungeons & Dragons (D&D) campaign, featuring majestic mountains rising from an island surrounded by ethereal clouds. The landscape should showcase vibrant shades of green representing lush forests and fields, as well as deep blue waters that encircle the island. The map should be depicted from a top-down perspective, allowing for a clear view of the terrain, topography, and any significant landmarks that may be present, such as rivers, valleys, or magical sites."

# strengths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
# strengths= [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# strengths = [0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]
strengths = [0.4]
fig, axs = plt.subplots(1, len(strengths), figsize=(30, 4))

for i, strength in enumerate(strengths):
    print(f"{i}/{len(strengths)-1}: strength {strength}")
    seed = random.randint(0, 1000000000)
    image = flux.generate_image(
        seed=seed,
        prompt=prompt,
        config=Config(
            num_inference_steps=20,
            height=1024,
            width=1024,
            init_image_path=image_path,
            init_image_strength=strength,
            guidance=2.0
        )
    )   
    # save the tmp image
    image.save(f"data/maps/flux_lora_dev/map_{i}.png")
    axs[i].imshow(image.image)
    axs[i].set_title(f"Strength: {strength}")
    axs[i].axis('off')

plt.tight_layout()
plt.savefig("data/maps/flux_lora_dev/map_forest_10000_2.png", dpi=300)