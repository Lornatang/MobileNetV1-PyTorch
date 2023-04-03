# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import queue##undefined
import sys
import threading##undefined
from glob import glob##undefined

import cv2##undefined
import torch
from PIL import Image##undefined
from torch.utils.data import Dataset, DataLoader##undefined
from torchvision import transforms##undefined
from torchvision.datasets.folder import find_classes##undefined
from torchvision.transforms import TrivialAugmentWide##undefined

import imgproc

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"


class ImageDataset(Dataset):##undefined
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mean: list, std: list, mode: str) -> None:
        super(ImageDataset, self).__init__()
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")##undefined
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)##undefined
        self.image_size = image_size##undefined
        self.mode = mode##undefined
        self.delimiter = delimiter

        if self.mode == "Train":##undefined
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                TrivialAugmentWide(),##undefined
                transforms.RandomRotation([0, 270]),##undefined
                transforms.RandomHorizontalFlip(0.5),##undefined
                transforms.RandomVerticalFlip(0.5),##undefined
            ])
        elif self.mode == "Valid" or self.mode == "Test":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop([self.image_size, self.image_size]),
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std)
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:##undefined
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]##undefined
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:##undefined
            image = cv2.imread(self.image_file_paths[batch_index])##undefined
            target = self.class_to_idx[image_dir]##undefined
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)##undefined

        # OpenCV convert PIL
        image = Image.fromarray(image) ##undefined

        # Data preprocess
        image = self.pre_transform(image) ##undefined

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False) ##undefined

        # Data postprocess
        tensor = self.post_transform(tensor) ##undefined

        return {"image": tensor, "target": target} ##undefined

    def __len__(self) -> int:
        return len(self.image_file_paths)


class PrefetchGenerator(threading.Thread): ##undefined
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:##undefined
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue) ##undefined
        self.generator = generator##undefined
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item) ##undefined
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):##undefined
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None: ##undefined
        self.num_data_prefetch_queue = num_data_prefetch_queue ##undefined
        super(PrefetchDataLoader, self).__init__(**kwargs) 

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:##undefined
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher: ##undefined
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): ##undefined
        self.batch_data = None ##undefined
        self.original_dataloader = dataloader ##undefined
        self.device = device ##undefined

        self.data = iter(dataloader) ##undefined
        self.stream = torch.cuda.Stream() ##undefined
        self.preload() ##undefined

    def preload(self):##undefined
        try: ##undefined
            self.batch_data = next(self.data) ##undefined
        except StopIteration: ##undefined
            self.batch_data = None ##undefined
            return None ##undefined

        with torch.cuda.stream(self.stream): ##undefined
            for k, v in self.batch_data.items(): ##undefined
                if torch.is_tensor(v): ##undefined
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) ##undefined

    def next(self):##undefined
        torch.cuda.current_stream().wait_stream(self.stream) ##undefined
        batch_data = self.batch_data ##undefined
        self.preload() ##undefined
        return batch_data ##undefined

    def reset(self):##undefined
        self.data = iter(self.original_dataloader) ##undefined
        self.preload() ##undefined

    def __len__(self) -> int:##undefined
        return len(self.original_dataloader) ##undefined
