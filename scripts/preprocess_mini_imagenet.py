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
import csv##used to import csv module which helps for reading and writing data from/to csv files
import os

from PIL import Image

train_csv_path = "../data/MiniImageNet_1K/original/train.csv"##assign to train_csv_path the relative path to the train.csv file
valid_csv_path = "../data/MiniImageNet_1K/original/valid.csv"##assign to valid_csv_path the relative path to the valid.csv file
test_csv_path = "../data/MiniImageNet_1K/original/test.csv"##assign to test_csv_path the relative path to the test.csv file

inputs_images_dir = "../data/MiniImageNet_1K/original/mini_imagenet/images"##assign to inputs_images_dir the relative path to the directory images that contains a set of images
output_images_dir = "../data/MiniImageNet_1K/"

train_label = {}##initialize train_label dictionary as empty
val_label = {}##initialize val_label dictionary as empty
test_label = {}##initialize test_label dictionary as empty
with open(train_csv_path) as csvfile:##open the csv file specified in the path assigned to train_csv_path
    csv_reader = csv.reader(csvfile)##creates csv_reader object using csv.reader() which reads data from csvfile
    birth_header = next(csv_reader)##reads the first row from the csv file and stores it in birth_header and based on the next, the next reading will be done from the next row after header
    for row in csv_reader:##iterates through the remaining rows from the csv file
        train_label[row[0]] = row[1]##add data to the dictionary train_label where the key is the first column ffrom each row and the value is the second column.

with open(valid_csv_path) as csvfile:##open the file specified in the path assigned to valid_csv_path
    csv_reader = csv.reader(csvfile)##creates csv_reader object using csv.reader() which reads data from csvfile
    birth_header = next(csv_reader)##reads the first row from the csv file and stores it in birth_header and based onthe next, the next reading will be done from the next row after header
    for row in csv_reader:##iterates through the remaining rows from the csv file
        val_label[row[0]] = row[1]##add data to the dictionary val_label where the key is the first column ffrom each row and the value is the second column.

with open(test_csv_path) as csvfile:##open the file specified in the path assigned to test_csv_path
    csv_reader = csv.reader(csvfile)##creates csv_reader object using csv.reader() which reads data from csvfile
    birth_header = next(csv_reader)##reads the first row from the csv file and stores it in birth_header and based onthe next, the next reading will be done from the next row after header
    for row in csv_reader:##iterates through the remaining rows from the csv file
        test_label[row[0]] = row[1]##add data to the dictionary test_label where the key is the first column ffrom each row and the value is the second column.

for png in os.listdir(inputs_images_dir):##iterates through each image from the directory situated at the relative path assign to inputs_images_dir
    path = inputs_images_dir + "/" + png##create the relative path for each image from the directory
    im = Image.open(path)##opens the image file situated at the created path and assign the resulting image to im
    if png in train_label.keys():##checks if the current image file is found as key in the train_label directory and if yes
        tmp = train_label[png]##gets the value of the png key from train_label directory
        temp_path = output_images_dir + "/train" + "/" + tmp##create a new relative path for the obtained value by concatenating output_images_dir with a subdirectory train and tmp
        if not os.path.exists(temp_path):##checks if the there exists something at the created path and if not
            os.makedirs(temp_path)##created a new directory specified by temp_path
        t = temp_path + "/" + png##creates the file path for the current image file where the image will be saved
        im.save(t)

    elif png in val_label.keys():##checks if the current image file name is found as key in the val_label directory and if yes
        tmp = val_label[png]##gets the value of the png key from val_label directory
        temp_path = output_images_dir + "/valid" + "/" + tmp ##create a new relative path for the obtained value by concatenating output_images_dir with a subdirectory valid and tmp
        if not os.path.exists(temp_path):##hecks if the there exists something at the created path
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t) ##save the current image file to the relative path created in t

    elif png in test_label.keys(): ##checks if the current image file name is found as key in the test_label directory and if yes
        tmp = test_label[png] ##gets the value of the png key from test_label directory
        temp_path = output_images_dir + "/test" + "/" + tmp ##create a new relative path for the obtained value by concatenating output_images_dir with a subdirectory test and tmp
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t)
