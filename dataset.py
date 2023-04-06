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
import queue#imports the queue data structure
import sys
import threading#imports multithreading
from glob import glob#imports the glob library

import cv2#imports the openCV library
import torch
from PIL import Image#from the Python Imaging Library imports Image
from torch.utils.data import Dataset, DataLoader#from pytorch imports Dataset, DataLoader
from torchvision import transforms# from torchvision imports transforms
from torchvision.datasets.folder import find_classes# from torchvision imports find_classes
from torchvision.transforms import TrivialAugmentWide# form torchvision imports TrivialAugmentWide

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


class ImageDataset(Dataset):# class definition of ImageDataset
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
        self.image_file_paths = glob(f"{image_dir}/*/*")#sets the path for the image files
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)#calls the find_classes() function on the image path to extract the classes
        self.image_size = image_size# sets the image size to the one provided in the constructor
        self.mode = mode# same, but with mode
        self.delimiter = delimiter

        if self.mode == "Train":# checks if it is train mode
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                TrivialAugmentWide(),#data augmentation function applied to the image data for transformation
                transforms.RandomRotation([0, 270]),#randomly rotates the image by an angle
                transforms.RandomHorizontalFlip(0.5),#randomly flips the image horizontally with probablity of 50%
                transforms.RandomVerticalFlip(0.5),#randomly flips the image vertically with probability of 50%
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

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:#this method allows objects to be accessed as lists/arrays
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]# fiile name and file path
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:#checks if the file extension is in the given library of accepted extensions
            image = cv2.imread(self.image_file_paths[batch_index])#reads the image from the path
            target = self.class_to_idx[image_dir]#looks up the target class index
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#converts the image from BGR to RGB

        # OpenCV convert PIL
        image = Image.fromarray(image) #constructs an image with Python Image Library

        # Data preprocess
        image = self.pre_transform(image) #applies the preprocessing method

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False) #converts the image to a tensor

        # Data postprocess
        tensor = self.post_transform(tensor) # applies the postprocessing method

        return {"image": tensor, "target": target} # returns the final tensor created with the target

    def __len__(self) -> int:
        return len(self.image_file_paths)


class PrefetchGenerator(threading.Thread): # class definition of PrefetchGenerator
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:#constructor definition wih data generator and how many early data load queues
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue) #creates a new queue with num_data_prefetch_queue size
        self.generator = generator# sets the generator to the one provided in the constructor
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item) #puts the current item in the queue
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):# class definition of the PrefetchDataLoader
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None: #constructor for the class
        self.num_data_prefetch_queue = num_data_prefetch_queue #sets the num_data_prefetch_queue to the one provided in the constructor
        super(PrefetchDataLoader, self).__init__(**kwargs) 

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:#class definition of the CPUPreefetcher
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


class CUDAPrefetcher: #class definition of the CUDAPrefetcher
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): #constructor definition
        self.batch_data = None #sets the batch data to null
        self.original_dataloader = dataloader #sets the original_dataloader to the one provided in the parameters
        self.device = device #same, buth with device

        self.data = iter(dataloader) #extracts the data from the dataloader in a variable 'data'
        self.stream = torch.cuda.Stream() #sets the CUDA stream
        self.preload() #calls the preload method

    def preload(self):#declaration of the preload method
        try: #try/catch structure start
            self.batch_data = next(self.data) #tries to extract the next data
        except StopIteration: #if no data received
            self.batch_data = None #set the data to null
            return None #return from the method

        with torch.cuda.stream(self.stream): #create a new cuda stream for the current device
            for k, v in self.batch_data.items(): # iterate over the key,value in the data items
                if torch.is_tensor(v): #checks whether the current value 'v' is a tensor
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) #moves the tensor value 'v' to the 'device', then gets assigned back to the original key 'k'

    def next(self):#the next method
        torch.cuda.current_stream().wait_stream(self.stream) #waits for the current stream
        batch_data = self.batch_data #sets the batch_data value
        self.preload() #calls preload method
        return batch_data #returns the batch data

    def reset(self):#reset method
        self.data = iter(self.original_dataloader) #sets the data to the original dataset
        self.preload() #calls preload method

    def __len__(self) -> int:#implements the len method
        return len(self.original_dataloader) #returns the length of the original dataset
