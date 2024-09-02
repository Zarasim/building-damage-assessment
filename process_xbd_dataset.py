import json
import shutil
from pathlib import Path
import random
from tqdm import tqdm
from shapely.geometry import Polygon


def extract_bbox_from_polygon(polygon_str):
    # Convert WKT string to list of coordinates
    coords = polygon_str.strip("POLYGON (())").split(", ")
    coords = [tuple(map(float, coord.split())) for coord in coords]

    # Create a Shapely polygon
    poly = Polygon(coords)

    # Get the bounding box
    minx, miny, maxx, maxy = poly.bounds

    return {
        'min_x': minx,
        'min_y': miny,
        'max_x': maxx,
        'max_y': maxy
    }


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert bounding box to YOLO format."""
    x_center = (bbox['min_x'] + bbox['max_x']) / 2 / img_width
    y_center = (bbox['min_y'] + bbox['max_y']) / 2 / img_height
    width = (bbox['max_x'] - bbox['min_x']) / img_width
    height = (bbox['max_y'] - bbox['min_y']) / img_height
    return x_center, y_center, width, height


def process_xbd_dataset(xbd_path, output_path, split_ratio=(0.7, 0.2, 0.1)):
    """Process xBD dataset and convert to YOLO format."""
    xbd_path = Path(xbd_path)
    output_path = Path(output_path)

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Define damage classes
    damage_classes = {
        'no-damage': 0,
        'minor-damage': 1,
        'major-damage': 2,
        'destroyed': 3
    }

    # Process each tier
    all_images = []

    images_dir = xbd_path / 'images'
    labels_dir = xbd_path / 'labels'

    # Process each image and its corresponding label
    for img_path in tqdm(list(images_dir.glob('*.png')), desc=f"Processing {images_dir.name}"):
        label_path = labels_dir / f"{img_path.stem}.json"

        if not label_path.exists():
            print(f"Warning: No label file for {img_path}")
            continue

        with open(label_path, 'r') as f:
            label_data = json.load(f)

        img_width = label_data['metadata']['width']
        img_height = label_data['metadata']['height']

        yolo_annotations = []
        for feature in label_data['features']['xy']:
            damage_class = damage_classes.get(feature['properties'].get('subtype', -1), -1)
            if damage_class == -1:
                continue

            bbox = extract_bbox_from_polygon(feature['wkt'])
            x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
            yolo_annotations.append(f"{damage_class} {x_center} {y_center} {width} {height}")

        if yolo_annotations != []:
            all_images.append((img_path, yolo_annotations))

    # Shuffle and split the dataset
    random.shuffle(all_images)
    train_split = int(len(all_images) * split_ratio[0])
    val_split = int(len(all_images) * (split_ratio[0] + split_ratio[1]))

    # Copy images and create YOLO label files
    for idx, (img_path, annotations) in enumerate(tqdm(all_images, desc="Copying files")):
        if idx < train_split:
            split = 'train'
        elif idx < val_split:
            split = 'val'
        else:
            split = 'test'

        # Copy image
        dest_img_path = output_path / 'images' / split / img_path.name
        shutil.copy(img_path, dest_img_path)

        # Create YOLO label file
        label_file_path = output_path / 'labels' / split / f"{img_path.stem}.txt"
        with open(label_file_path, 'w') as f:
            f.write('\n'.join(annotations))

    # Create data.yaml file
    data_yaml_content = f"""
    train: {output_path}/images/train
    val: {output_path}/images/val
    test: {output_path}/images/test

    nc: 4
    names: ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
        """
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(data_yaml_content.strip())

    print(f"Dataset processed and saved to {output_path}")


# Usage
xbd_dataset_path = "/home/simone/github/building-damage-assessment/train"
output_dataset_path = "/home/simone/github/building-damage-assessment/yolo/dataset"
process_xbd_dataset(xbd_dataset_path, output_dataset_path)
