
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

pool = ThreadPoolExecutor(max_workers=20)

def test_file(image_path):
    try:
        image = Image.open(image_path)    
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image, np.uint8)
    except Exception as e:
        print(e)
        print(image_path)

for image_path in tqdm(Path('/fsx/data/stable_diffusion/ft_humans/getty_images/').glob('*.jpg')):
    pool.submit(test_file, image_path)

pool.shutdown()