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
import queue##import queue module
import sys
import threading##import threading module in order to allow to work with threads
from glob import glob##import glob class from glob module, which is used to search for files and directories whose names match a specified pattern.

import cv2##import cv2 in order to be able to work with openCV
import torch
from PIL import Image##import Image class from PIL modul, which provides a collection of classes for opening, manipulating, and saving many different image file formats.
from torch.utils.data import Dataset, DataLoader##import Dataset and DataLoader from torch.utils.data module. These classes are used to handle and load data in PyTorch.
from torchvision import transforms##import transforms from torchvision module, provides a set of common image transformations, such as resizing, cropping, and normalization.
from torchvision.datasets.folder import find_classes##import find_classes from torchvision.datasets.folder module,  used to find the class names of a dataset, which is important when training a model to classify images into different categories.
from torchvision.transforms import TrivialAugmentWide##import TrivialAugmentWide from torchvision.transforms module, applies a set of image augmentations, such as random flipping and rotation, to the input images to increase the diversity of the training data and improve the robustness of the trained model.

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


class ImageDataset(Dataset):##define a new class ImageDataset which inherits from Dataset class, which is used to load and preprocess image data for training or testing machine learning models.
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
        self.image_file_paths = glob(f"{image_dir}/*/*")##this uses glob function to get the file paths of all the images in the image_dir directory
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)##this uses find_classes function to get a list of class names and their corresponding indices based on the directory structure. The first returned value is ignored because of the _
        self.image_size = image_size## sets the size of the images to be loaded
        self.mode = mode##sets the mode of the data loading, which can be either "Train", "Valid", or "Test"
        self.delimiter = delimiter

        if self.mode == "Train":##if mode is Train
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                TrivialAugmentWide(),##this is a custom transformation used to wide the image
                transforms.RandomRotation([0, 270]),##randomly rotate the image by a random angle between 0 and 270 degrees
                transforms.RandomHorizontalFlip(0.5),##randomly flip the image horizontally with a probability of 0.5
                transforms.RandomVerticalFlip(0.5),##randomly flip the image vertically with a probability of 0.5
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

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:##the method takes an index of the batch that we want to access, and returns a tuple containing the preprocessed image data and its corresponding label.
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]##selects the file path at the index batch_index from the list and splits it into parts using the split method, with self.delimiter as the separator. The [-2:] indexing takes the last two elements of the list which could be the name of the subdirectory containing the image and the name of the image file
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:##verifica faca extensia imaginii se gaseste in lista de extensii
            image = cv2.imread(self.image_file_paths[batch_index])##read the image file using openCV function and assign it to image
            target = self.class_to_idx[image_dir]##xtracts the class label of the image from the class_to_idx dictionary using the image_dir variable, which is the directory name that contains the image file
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)##converts the image from BGR to RGB

        # OpenCV convert PIL
        image = Image.fromarray(image) ##converts the image from a numpy array to a PIL (Python Imaging Library) image

        # Data preprocess
        image = self.pre_transform(image) ##applies a set of image transformations to the image using the pre_transform object that was defined earlier

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False) ##converts the PIL image to a PyTorch tensor

        # Data postprocess
        tensor = self.post_transform(tensor) ##applies a normalization transformation to the tensor using the post_transform object that was defined earlier

        return {"image": tensor, "target": target} ##returns a dictionary containing the tensor and the target class label

    def __len__(self) -> int:
        return len(self.image_file_paths)


class PrefetchGenerator(threading.Thread): ##PrefetchGenerator that inherits from threading.Thread, which means it is a subclass of the Thread class in the threading module
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:##the constructor of the PrefetchGenerator class, which takes in two parameters: generator, which is a data generator object that generates batches of data, and num_data_prefetch_queue, which is an integer that determines the size of a queue used for data prefetching
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue) ## is creating a queue object of size num_data_prefetch_queue
        self.generator = generator##is assigning the generator object passed as a parameter to the generator attribute of the object
        self.daemon = True
        self.start()

    def run(self) -> None: # is starting the thread by calling the start method of the object
        for item in self.generator:
            self.queue.put(item) ##puts each generated batch of data into the queue
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):## define PrefetchDataLoader class, which is a subclass of the DataLoader class provided by PyTorch
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None: ##this is the constructor which takes 2 arguments, num_data_prefetch_queue which specifies how many early data load queues should be used, **kwargs is a catch-all keyword arguments parameter that allows the constructor to accept any number of additional parameters
        self.num_data_prefetch_queue = num_data_prefetch_queue ##intialise num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs) 

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:##CPUPrefetcher class is a data loader that prefetches data from the CPU to the GPU in a separate thread. It is designed to speed up the data loading process for deep learning models by asynchronously loading the next batch of data while the current batch is being processed by the GPU.
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


class CUDAPrefetcher: ##define class named CUDAPrefetcher to indicate that CUDA is used
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): ##the CUDAPrefetcher class is being initialized with three arguments: dataloader, which is an instance of the DataLoader class, device, which is an instance of the torch.device class, and self, which is a reference to the newly created instance of the CUDAPrefetcher class
        self.batch_data = None ##batch_data set to None, batch_data will be used to store the next batch of data to be processed by the GPU
        self.original_dataloader = dataloader ##initialise original_dataloader to dataloader, original_dataloader will be used later to reset the CUDAPrefetcher to its initial state
        self.device = device ##initialise device, device will be used to specify the device (CPU or GPU) that the data should be processed on

        self.data = iter(dataloader) ##This initializes the data attribute of the CUDAPrefetcher instance to an iterator created from the dataloader argument data will be used to prefetch the next batch of data to be processed by the GPU
        self.stream = torch.cuda.Stream() ##initialise stream, stream will be used to synchronize the GPU operations
        self.preload() ##load the first batch of data to be processed by the GPU

    def preload(self):##define the preload method which is called in the constructor
        try: ##starts a try block in order to catch the possible excpetions
            self.batch_data = next(self.data) ##get the next batch of data from the original data loader
        except StopIteration: ## if a StopIteration exception is raised when trying to get the next batch of data the below lines will be executed
            self.batch_data = None ##set batch_data to None
            return None ##return None

        with torch.cuda.stream(self.stream): ##Creates a new CUDA stream to asynchronously transfer data
            for k, v in self.batch_data.items(): ## Loops over each key-value pair in the batch data dictionary
                if torch.is_tensor(v): ##Checks whether the value v is a PyTorch tensor
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) ##it is transferred to the specified device asynchronously with the to method. The non_blocking=True argument makes the transfer asynchronous. The modified tensor is then assigned back to the batch_data dictionary under the same key k

        def next(self):##define the next method
            torch.cuda.current_stream().wait_stream(self.stream) ## waits for all previous operations in the current stream to complete before executing the next line
            batch_data = self.batch_data ##assigns the current batch data to a local variable called batch_data
            self.preload() ##preloads the next batch of data from the data loader and performs CUDA memory transfers as necessary
            return batch_data ##return the batch data

    def reset(self):##define reset method which reset the state of the object to its initial configuration
        self.data = iter(self.original_dataloader) ##iter() function is a built-in Python function that returns an iterator object, which can be used to traverse a sequence of elements one by one
        self.preload() ##call preload function

    def __len__(self) -> int:##define the method wich return the length of an object
        return len(self.original_dataloader) ##return the length of the original dataloader
