# import the json in output.json

import json
from Prepare_dataset import Prepare_dataset 
import cv2
import os

with open('output.json') as f:
    data = json.load(f)
    
print("Number of videos: ", len(data))
print("Number of Segments: ", sum([len(data[i]["clips"]) for i in range(len(data))]))

print(data[0]["clips"][0]['features'])
# data_driver = Prepare_dataset()

# data_driver.get_annotations(data[0]["video"],1)

# print("Number of frames: ", (data[0]["clips"][-1]["segment"][-1]))

# data_driver.get_annotations(data[1]["video"],1)

# print("Number of frames: ", (data[1]["clips"][-1]["segment"][-1]))

# data_driver.get_annotations(data[2]["video"],1)

# print("Number of frames: ", (data[2]["clips"][-1]["segment"][-1]))

