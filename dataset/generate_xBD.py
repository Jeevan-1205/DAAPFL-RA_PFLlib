import numpy as np
import os
import re
import glob
import random
from PIL import Image
from utils.dataset_utils import split_data, save_file


random.seed(1)
np.random.seed(1)
num_classes = 5  # 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
img_size = 224
dir_path = "xBD/"
data_path = "/DATA/BU_Internship_Jeevan/Research/data/xbd_full/geotiffs/"

# Disaster types used as federated clients (one client per type)
disaster_types = [
    'Earthquake', 'Flood', 'Hurricane', 'Tornado',
    'Tsunami', 'Volcano', 'Wildfire',
]


def get_disaster_type(event_name):
    """Map a specific xBD event name to its disaster type."""
    name = event_name.lower()
    if 'hurricane' in name:
        return 'Hurricane'
    elif 'earthquake' in name:
        return 'Earthquake'
    elif 'flood' in name:
        return 'Flood'
    elif 'tornado' in name:
        return 'Tornado'
    elif 'fire' in name or 'bushfire' in name:
        return 'Wildfire'
    elif 'tsunami' in name:
        return 'Tsunami'
    elif 'volcano' in name:
        return 'Volcano'
    return 'Unknown'


def scan_xbd_samples(data_path, splits):
    """Scan all splits and return list of (pre_path, post_path, mask_path, event)."""
    samples = []
    for split in splits:
        img_dir = os.path.join(data_path, split, 'images')
        mask_dir = os.path.join(data_path, split, 'masks')
        if not os.path.isdir(img_dir):
            continue

        pre_images = sorted(glob.glob(os.path.join(img_dir, '*_pre_disaster.tif')))
        for pre_path in pre_images:
            base = os.path.basename(pre_path)
            event = re.sub(r'_\d+_pre_disaster\.tif$', '', base)
            post_path = pre_path.replace('_pre_disaster.tif', '_post_disaster.tif')
            mask_path = os.path.join(
                mask_dir,
                base.replace('_pre_disaster.tif', '_post_disaster.png')
            )

            if os.path.exists(post_path) and os.path.exists(mask_path):
                samples.append((pre_path, post_path, mask_path, event))

    return samples


def load_and_resize_mask(mask_path, size):
    """Read a PNG mask, resize with nearest interpolation, return (H,W) uint8."""
    mask = np.array(Image.open(mask_path))
    mask = np.array(Image.fromarray(mask).resize((size, size), Image.NEAREST))
    assert mask.min() >= 0
    assert mask.max() <= 4
    return mask



# Allocate data to users
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get xBD data
    splits = ['tier1', 'tier3', 'test', 'hold']
    all_samples = scan_xbd_samples(data_path, splits)
    print(f'Total image pairs found: {len(all_samples)}')

    # Group samples by disaster type
    type_to_samples = {dt: [] for dt in disaster_types}
    for sample in all_samples:
        dt = get_disaster_type(sample[3])
        if dt in type_to_samples:
            type_to_samples[dt].append(sample)

    X, y = [], []
    for dt in disaster_types:
        samples = type_to_samples[dt]
        print(f'\nIndexing {dt}: {len(samples)} image pairs')

        dataset_image = [(pre_path, post_path) for pre_path, post_path, _, _ in samples]
        dataset_label = [mask_path for _, _, mask_path, _ in samples]

        X.append(np.array(dataset_image, dtype=object))  # (N, 2) paths: pre, post
        y.append(np.array(dataset_label, dtype=object))  # (N,) paths: masks

    num_clients = len(y)
    print(f'\nNumber of clients: {num_clients}')
    print(f'Number of classes: {num_classes}')

    # Compute per-client statistics (pixel-level class counts)
    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        class_counts = np.zeros(num_classes, dtype=np.int64)
        for mask_path in y[client]:
            mask = load_and_resize_mask(mask_path, img_size)
            counts = np.bincount(mask.flatten(), minlength=num_classes)
            class_counts += counts[:num_classes]
        for c, count in enumerate(class_counts):
            if count > 0:
                statistic[client].append((int(c), int(count)))

    for client in range(num_clients):
        labels = [label for label, _ in statistic[client]]
        print(f"Client {client} ({disaster_types[client]})\t Size of data: {len(X[client])}\t Labels: ", labels)
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, None, None, None)


if __name__ == "__main__":
    generate_dataset(dir_path)
