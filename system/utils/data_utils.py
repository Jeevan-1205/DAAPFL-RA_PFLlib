import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import time


XBD_IMG_SIZE = 224
DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"


def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = DATASET_ROOT / dataset / 'train'
    else:
        data_dir = DATASET_ROOT / dataset / 'test'

    file = data_dir / f'{idx}.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data
# /DATA/BU_Internship_Jeevan/Research/PFLlib/system/utils/data_utils.py

def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    elif "xbd" in dataset.lower():
        data_list = process_xbd(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0 and "xbd" not in dataset.lower():
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


class XBDDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=XBD_IMG_SIZE):
        self.image_paths = [self._to_pair(paths) for paths in image_paths]
        self.mask_paths = [str(path) for path in mask_paths]
        self.image_size = image_size

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return XBDDataset(
                self.image_paths[index],
                self.mask_paths[index],
                image_size=self.image_size,
            )

        if isinstance(index, (list, tuple, np.ndarray)):
            indices = np.asarray(index).tolist()
            return XBDDataset(
                [self.image_paths[i] for i in indices],
                [self.mask_paths[i] for i in indices],
                image_size=self.image_size,
            )

        pre_path, post_path = self.image_paths[index]
        pre = self._load_tif(pre_path)
        post = self._load_tif(post_path)
        mask = self._load_mask(self.mask_paths[index])

        stacked = np.concatenate([pre, post], axis=2).transpose(2, 0, 1)
        x = torch.from_numpy(stacked.copy()).type(torch.float32)
        y = torch.from_numpy(mask.copy()).type(torch.int64)
        return x, y

    @staticmethod
    def _to_pair(paths):
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()
        if len(paths) != 2:
            raise ValueError(f"xBD samples must contain pre/post image paths, got {paths}")
        return str(paths[0]), str(paths[1])

    def _load_tif(self, tif_path):
        import tifffile

        image = tifffile.imread(tif_path).astype(np.float32)
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min + 1e-6)
        image = (image * 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[-1] > 3:
            image = image[:, :, :3]

        image = Image.fromarray(image).resize(
            (self.image_size, self.image_size),
            _pil_resampling("bilinear"),
        )
        return np.asarray(image, dtype=np.float32) / 255.0

    def _load_mask(self, mask_path):
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = Image.fromarray(mask).resize(
            (self.image_size, self.image_size),
            _pil_resampling("nearest"),
        )
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.min() < 0 or mask.max() > 4:
            raise ValueError(f"xBD mask labels must be in [0, 4], got {mask_path}")
        return mask

class XBDCachedDataset(Dataset):
    """Reads precomputed .npz tiles built by scripts/prepare_xbd_cache.py.
    No TIFF decode, no resize, no min/max scan at runtime — just np.load."""

    def __init__(self, image_paths, mask_paths, image_size=XBD_IMG_SIZE, cache_root=None):
        self.image_paths = [self._to_pair(p) for p in image_paths]
        self.image_size = image_size
        self.cache_root = Path(cache_root) if cache_root else (
            DATASET_ROOT / "xBD" / "tile_cache"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return XBDCachedDataset(self.image_paths[index], None,
                                     self.image_size, self.cache_root)
        if isinstance(index, (list, tuple, np.ndarray)):
            indices = np.asarray(index).tolist()
            return XBDCachedDataset([self.image_paths[i] for i in indices],
                                     None, self.image_size, self.cache_root)

        pre_path, post_path = self.image_paths[index]
        cache_file = self._cache_path(pre_path, post_path)

        with np.load(cache_file) as z:
            stacked = z["image"]   # (H, W, 6) float32, already resized+normalized
            mask = z["mask"]       # (H, W) uint8, already resized

        x = torch.from_numpy(stacked.transpose(2, 0, 1).copy()).type(torch.float32)
        y = torch.from_numpy(mask.copy()).type(torch.int64)
        return x, y

    def _cache_path(self, pre_path, post_path):
        stem = Path(pre_path).stem + "__" + Path(post_path).stem
        return self.cache_root / f"{stem}.npz"

    @staticmethod
    def _to_pair(paths):
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()
        if len(paths) != 2:
            raise ValueError(f"xBD samples must contain pre/post image paths, got {paths}")
        return str(paths[0]), str(paths[1])

    def _load_tif(self, tif_path):
        import tifffile

        image = tifffile.imread(tif_path).astype(np.float32)
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min + 1e-6)
        image = (image * 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[-1] > 3:
            image = image[:, :, :3]

        image = Image.fromarray(image).resize(
            (self.image_size, self.image_size),
            _pil_resampling("bilinear"),
        )
        return np.asarray(image, dtype=np.float32) / 255.0

    def _load_mask(self, mask_path):
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = Image.fromarray(mask).resize(
            (self.image_size, self.image_size),
            _pil_resampling("nearest"),
        )
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.min() < 0 or mask.max() > 4:
            raise ValueError(f"xBD mask labels must be in [0, 4], got {mask_path}")
        return mask


def _pil_resampling(mode):
    if hasattr(Image, "Resampling"):
        if mode == "bilinear":
            return Image.Resampling.BILINEAR
        if mode == "nearest":
            return Image.Resampling.NEAREST
    if mode == "bilinear":
        return Image.BILINEAR
    if mode == "nearest":
        return Image.NEAREST
    raise ValueError(f"Unsupported PIL resampling mode: {mode}")


def process_xbd(data):
    return XBDCachedDataset(data['x'], data['y'])


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]
