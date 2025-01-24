from diffusers.utils import load_image
from diffusers import FluxImg2ImgPipeline
import matplotlib.pyplot as plt
import torch
import os

#  Config
HUGGINGFACE_HUB_CACHE="${SCRATCH}/huggingface_cache"
device = "cuda"
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load the model
pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir=HUGGINGFACE_HUB_CACHE).to(device)


# lora_paths = os.path.join(project_root, "loras", "lora_dev_10000.safetensors")
# lora_scales=[1.0]
# pipe.load_lora_weights(lora_paths)

prompt = "Create a detailed fantasy map for a Dungeons & Dragons (D&D) campaign, featuring majestic mountains rising from an island surrounded by ethereal clouds. The map should be depicted from a top-down perspective, moody detailed terrain, and any significant landmarks that may be present, such as rivers, homes, towers, forests, and othes structures."


styles = ["forest", "desert", "dark", "ocean"]
strengths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
guidance_scales = [3, 4, 5, 6, 7, 8, 9, 10]


for style in styles:
    print(f"\n\n\nGenerating map for style: {style}\n\n\n")
    image_path = os.path.join(project_root, "data", "maps", f"map_{style}.png")

    init_image = load_image(image_path).resize((1024, 1024))
    fig, axs = plt.subplots(len(strengths), len(guidance_scales), figsize=(40, 20))

    for i, strength in enumerate(strengths):
        for j, guidance_scale in enumerate(guidance_scales):
            print(f"\nStrength: {strength}, Guidance: {guidance_scale}")
            images = pipe(
                prompt=prompt, 
                image=init_image,
                num_inference_steps=30, 
                strength=strength, 
                guidance_scale=guidance_scale
            ).images[0]
            axs[i, j].imshow(images)
            axs[i, j].set_title(f"Strength: {strength}, Guidance Scale: {guidance_scale}", fontsize=8)
            axs[i, j].axis('off')
            # save the image
            images.save(f"data/maps/flux_dev_base/map_{style}_{strength}_{guidance_scale}.png")

    plt.tight_layout()
    plt.savefig(f"results/base/result_map_{style}.png", dpi=300)