from math import ceil
import os
import random
import csv
from opts import parse_opts
from math import ceil

opt = parse_opts()
path = opt.video_root
annotations_path = opt.annotation_path

class Prepare_dataset:
    def __init__(self, mode = "training", volume = 1, frames = 15):
        self.path = path
        self.files = os.listdir(path)
        self.mode = mode
        self.volume = volume
        self.frames = frames
        
    def get_annotations(self, video):
        with open(annotations_path, "r") as f:
            reader = csv.reader(f, delimiter = "\t")
            for row in reader:
                if row[0]+".mp4" == video:
                    importance = [int(i) for i in row[2].split(",")]
                    print("len(importance): ", len(importance))
                    # get the average of each <frames * 3532 / 19395> frames, round up to the nearest integer
                    count = self.frames
                    segments = ceil(len(importance) / count)
                    segment_indices = [[ceil(i * count), min(ceil((i + 1) * count), len(importance))] for i in range(segments)]
                    importance = [sum(importance[i[0]:i[1]]) / (i[1] - i[0]) if i[1] != i[0] else importance[i[1]] if len(importance) > i[1] else 0 for i in segment_indices]
                    return importance
                
    def __call__(self):
        random.seed(0)
        random.shuffle(self.files)
        
        if self.mode == "training":
            files = self.files[:int(len(self.files) * self.volume)]
        else:
            files = self.files[int(len(self.files) * self.volume):]
        
        with open("input", "w") as f:
            for file in files:
                f.write(file + "\n")