# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine.data.dataset import COCO, Objects365, PascalVOC
from detection.tools.dataset import CustomerDataSet

data_mapper = dict(
    coco=COCO,
    cocomini=COCO,
    objects365=Objects365,
    voc=PascalVOC,
    chongqigongmen=CustomerDataSet,
)
