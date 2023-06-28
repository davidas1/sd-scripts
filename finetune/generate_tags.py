from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm

executor = ThreadPoolExecutor()

folder_path = Path('/fsx/data/fashion/generative-scraping-2022/getty_after_purchase/')
output_folder = Path('/fsx/data/stable_diffusion/ft_humans/getty_images/')
output_folder.mkdir(exist_ok=True)

def copy_image(src):
    if not (output_folder / src.name).exists():
        shutil.copy2(src, output_folder)

folders = list(folder_path.iterdir())
for sub_path in tqdm(folders, total=len(folders)):
    for img_path in sub_path.glob('*.jpg'):
        file_name = img_path.stem
        md_file = img_path.with_suffix('.yml')
        caption_path = output_folder / f'{file_name}.caption'
        tags_path = output_folder / f'{file_name}.txt'

        if not md_file.exists():
            continue

        executor.submit(copy_image, img_path)
        
        if caption_path.exists() and tags_path.exists():
            continue
        
        with open(md_file) as f:
            md = yaml.load(f,yaml.SafeLoader)
        caption = md.get('caption')
        if caption is None:
            caption = md.get('title')
        tags = [v['tag'] for v in md['tags']]
        if not caption_path.exists():
            with open(caption_path, 'w') as f:
                f.write(caption)
        if not tags_path.exists():
            tag_str = ''
            for i, tag in enumerate(tags):
                tag_str += tag
                if i+1 < len(tags):
                    tag_str += ', '
            with open(tags_path, 'w') as f:
                f.write(tag_str)

executor.shutdown()
print('DONE')