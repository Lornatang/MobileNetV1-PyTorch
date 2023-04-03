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
import csv##undefined
import os

from PIL import Image

train_csv_path = "../data/MiniImageNet_1K/original/train.csv"##undefined
valid_csv_path = "../data/MiniImageNet_1K/original/valid.csv"##undefined
test_csv_path = "../data/MiniImageNet_1K/original/test.csv"##undefined

inputs_images_dir = "../data/MiniImageNet_1K/original/mini_imagenet/images"##undefined
output_images_dir = "../data/MiniImageNet_1K/"

train_label = {}##undefined
val_label = {}##undefined
test_label = {}##undefined
with open(train_csv_path) as csvfile:##undefined
    csv_reader = csv.reader(csvfile)##undefined
    birth_header = next(csv_reader)##undefined
    for row in csv_reader:##undefined
        train_label[row[0]] = row[1]##undefined

with open(valid_csv_path) as csvfile:##undefined
    csv_reader = csv.reader(csvfile)##undefined
    birth_header = next(csv_reader)##undefined
    for row in csv_reader:##undefined
        val_label[row[0]] = row[1]##undefined

with open(test_csv_path) as csvfile:##undefined
    csv_reader = csv.reader(csvfile)##undefined
    birth_header = next(csv_reader)##undefined
    for row in csv_reader:##undefined
        test_label[row[0]] = row[1]##undefined

for png in os.listdir(inputs_images_dir):##undefined
    path = inputs_images_dir + "/" + png##undefined
    im = Image.open(path)##undefined
    if png in train_label.keys():##undefined
        tmp = train_label[png]##undefined
        temp_path = output_images_dir + "/train" + "/" + tmp##undefined
        if not os.path.exists(temp_path):##undefined
            os.makedirs(temp_path)##undefined
        t = temp_path + "/" + png##undefined
        im.save(t)

    elif png in val_label.keys():##undefined
        tmp = val_label[png]##undefined
        temp_path = output_images_dir + "/valid" + "/" + tmp ##undefined
        if not os.path.exists(temp_path):##undefined
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t) ##undefined

    elif png in test_label.keys(): ##undefined
        tmp = test_label[png] ##undefined
        temp_path = output_images_dir + "/test" + "/" + tmp ##undefined
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t)
