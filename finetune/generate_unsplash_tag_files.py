from pathlib import Path
import pandas as pd
from tqdm import tqdm

tags_out_path = Path('/fsx/data/stable_diffusion/ft_humans/tags_combined.csv')
tags_df = pd.read_csv(tags_out_path, converters={'tags': eval})
output_folder = Path('/fsx/data/stable_diffusion/ft_humans/images/')

for i, row in tqdm(tags_df.iterrows(), total=len(tags_df)):
    file_name = row['image_url'].split('/')[-1]
    tags = row['tags']
    tags_path = output_folder / f'{file_name}.txt'
    tag_str = ''
    for i, tag in enumerate(tags):
        tag_str += tag.replace('-', ' ')
        if i+1 < len(tags):
            tag_str += ', '
    with open(tags_path, 'w') as f:
        f.write(tag_str)

