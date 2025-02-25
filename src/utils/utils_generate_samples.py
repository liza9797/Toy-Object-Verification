import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw

def generate_shape_image(image_size=(32, 32), shape=None, s=None, min_size=5, 
                         position=None, angle=None):
    """
    Generates an image of a given size with shape, 
    random size, position and rotation.
    
    Params:
    - image_size: size of the image.
    - shape: 'circle' or 'square'. If None, generates randomly.
    - s: size of the shape. If None, it is selected randomly 
         from range [min_size, min(image_size)].
    - min_size: min size of shape for pandom generation.
    - position: (int, int) or None.
    - angle: angle of shape rotation for square shape. 
         If None, randomly select the angle of rotation from range [0, 90).
         Applyed only for shapes of the size > 6.
    
    Returns:
    - PIL.Image object.
    """
    
    img = Image.new("L", image_size, "white")
    draw = ImageDraw.Draw(img)
    
    ### Shape
    if shape is None or shape not in ["circle", "square"]:
        shape = random.choice(["circle", "square"])
    
    ### Size of the shape
    max_possible_size = min(image_size) 
    if s is None:
        s = random.randint(min_size, max_possible_size)
    
    ### Position of the shape
    if position is None:
        max_x = image_size[0] - s
        max_y = image_size[1] - s

        top_left_x = random.randint(0, max_x) if max_x > 0 else 0
        top_left_y = random.randint(0, max_y) if max_y > 0 else 0
    else:
        x, y = position
        top_left_x = x - s // 2
        top_left_y = y - s // 2
    bbox = [top_left_x, top_left_y, top_left_x + s, top_left_y + s]
    
    ### Draw
    if shape == "circle":
        draw.ellipse(bbox, fill="black")
    elif shape == "square":
        draw.rectangle(bbox, fill="black")
        
        ### Rotate square
        if s >= 6:
            if angle is None:
                angle = random.randint(0, 90)
#             img = img.rotate(angle, resample=Image.BILINEAR, fillcolor="white")
            img = img.rotate(angle, resample=Image.NEAREST, fillcolor="white")
    
    return img, shape


def generate_samples_dataframe():
    MIN_SIZE, MAX_SIZE = 3, 32
    IMG_SIZE = 32
    
    def generate_rows():
        for size in range(MIN_SIZE, MAX_SIZE + 1):
            for x in range(size//2, IMG_SIZE - size//2 + 1):
                for y in range(size//2, IMG_SIZE - size//2 + 1):
                    for shape in ["circle", "square"]:
                        for angle in range(0, 90, 3):
                            yield (shape, size, x, y, angle)
                            if (shape == "circle") or (size < 6):
                                break

    rows = [(i, *row) for i, row in enumerate(tqdm(list(generate_rows())))]

    columns = ['id', 'shape', 'size', 'x', 'y', 'angle']
    return pd.DataFrame(rows, columns=columns)

def generate_pairs(df1, df2, size, seed):
    sample1 = df1.sample(n=size, random_state=seed, replace=True).index
    sample2 = df2.sample(n=size, random_state=seed, replace=True).index
    return list(zip(sample1, sample2))

def create_train_val_test_split(df, seed=41):
    ### Train split
    for shape in ["circle", "square"]:
        train_samples = df[df['shape'] == shape].sample(n=5000, random_state=seed, replace=False)
        df.loc[train_samples.index, 'split'] = 'train'

    ### Val spit
    remaining = df.drop(df[df.split == 'train'].index)
    for shape in ["circle", "square"]:
        val_samples = remaining[remaining['shape'] == shape].sample(n=2000, random_state=seed, replace=False)
        df.loc[val_samples.index, 'split'] = 'val'

    ### Test split
    df.loc[df['split'].isna(), 'split'] = 'test'
    return df

