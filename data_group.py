#!/usr/bin/env python3
#gruppe2 train dataset
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from IPython import embed

json_path = 'chongqigongmen/data_label/job-403751-4.json'
train_anno_path = 'chongqigongmen/data_label/train_arch.json'
test_anno_path = 'chongqigongmen/data_label/test_arch.json'

def cvt_rect_json(p, w, h):
    return [
        int(p[0][0] * w),
        int(p[0][1] * h),
        int((p[1][0] - p[0][0]) * w),
        int((p[2][1] - p[1][1]) * h)
    ]


jsonfile_train = {}
jsonfile_test = {}

box_num = 1
p_num = 0
n_num = 0

# nw = nori.open("./inflatable_coco_arch.nori", "w")

with open(json_path, 'r') as f:
    file = json.load(f)
    images_train = []
    images_test = []
    annotations_train = []
    annotations_test = []
    idx = 0
    for item_id, item in enumerate(file['items']):
        file_name = os.path.join('chongqigongmen/data_label',
                                 item['resources'][0]['s'])
#         print(file_name)
        if item_id < 5000:
            images = images_train
            annotations = annotations_train
        else:
            images = images_test
            annotations = annotations_test

#         print(file_name)
        img = cv2.imread(file_name)
#         print(item["resources"][0]["s"].split("/")[-1])
        if img is None:
            continue
        else:
            image = {}
            image['id'] = idx
            image['width'] = item['resources'][0]['size']['width']
            image['height'] = item['resources'][0]['size']['height']
            image['file_name'] = item["resources"][0]["s"].split("/")[-1]
            images.append(image)
            idx += 1
            if item['results_state'] == 'ok':
                p_num = p_num + 1
                for box in item['results']['rects']:
                    annotation = {}
                    annotation['id'] = box_num
                    box_num = box_num + 1
                    annotation['image_id'] = image['id']
                    annotation['category_id'] = 0
                    rect = cvt_rect_json(box['rect'], image['width'], image['height'])
                    if rect[3] < 10 or rect[2] < 10:
                        continue
                    assert len(rect) == 4
                    annotation['bbox'] = rect
                    annotation['area'] = rect[2] * rect[3]
                    annotation['iscrowd'] = 0
                    annotations.append(annotation)
            else:
                n_num = n_num + 1
#info
jsonfile_train['info'] = 'Gruppe2 Arch Dataset for training'
jsonfile_test['info'] = 'Gruppe2 Arch Dataset for training'
#licenses
license = {}
license['id'] = 998
license['name'] = 'a drive license'
license['url'] = 'www.xxx.de'
licenses = []
licenses.append(license)
jsonfile_train['licenses'] = licenses
jsonfile_test['licenses'] = licenses
#images
jsonfile_train['images'] = images_train
jsonfile_test['images'] = images_test

#annotations
jsonfile_train['annotations'] = annotations_train
jsonfile_test['annotations'] = annotations_test

#categories
categories = []
category = {}
category['id'] = 0
category['name'] = 'arch'
category['supercategory'] = 'outdoor'
categories.append(category)
jsonfile_train['categories'] = categories
jsonfile_test['categories'] = categories

# print(jsonfile)
out_file = open(train_anno_path, "w")
json.dump(jsonfile_train, out_file)
out_file = open(test_anno_path, "w")
json.dump(jsonfile_test, out_file)
out_file.close()
# print('total imgs: ', len(os.listdir('/data/datasets/ArchData/ai-image/')))
print('train dataset: ', p_num + n_num)
print('positive data: ', p_num)
print('negative data: ', n_num)

# vim: ts=4 sw=4 sts=4 expandtab
