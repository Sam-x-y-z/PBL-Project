import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from Prepare_dataset import Prepare_dataset
import cv2
import shutil

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 15
    opt.n_classes = 400
    
    data_drive = Prepare_dataset()
    data_drive()
    
    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model,map_location=torch.device('cpu'))
    assert opt.arch == model_data['arch']
    # keys_to_be_deleted = ["module.layer1.0.downsample.0.weight", "module.layer1.0.downsample.1.weight", "module.layer1.0.downsample.1.bias", "module.layer1.0.downsample.1.running_mean", "module.layer1.0.downsample.1.running_var", "module.layer2.0.downsample.0.weight", "module.layer2.0.downsample.1.weight", "module.layer2.0.downsample.1.bias", "module.layer2.0.downsample.1.running_mean", "module.layer2.0.downsample.1.running_var", "module.layer3.0.downsample.0.weight", "module.layer3.0.downsample.1.weight", "module.layer3.0.downsample.1.bias", "module.layer3.0.downsample.1.running_mean", "module.layer3.0.downsample.1.running_var", "module.layer4.0.downsample.0.weight", "module.layer4.0.downsample.1.weight", "module.layer4.0.downsample.1.bias", "module.layer4.0.downsample.1.running_mean", "module.layer4.0.downsample.1.running_var"]
    # state_dict = model_data['state_dict']
    # modified_state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_be_deleted}
    model.load_state_dict({k[7:]:v for k, v in model_data['state_dict'].items() if k[7:] not in ["layer1.0.downsample.0.weight", "layer1.0.downsample.1.weight", "layer1.0.downsample.1.bias", "layer1.0.downsample.1.running_mean", "layer1.0.downsample.1.running_var", "layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer4.0.downsample.0.weight", "layer4.0.downsample.1.weight", "layer4.0.downsample.1.bias", "layer4.0.downsample.1.running_mean", "layer4.0.downsample.1.running_var"]})
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        os.removedirs('tmp')

    outputs = []
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            print(video_path)
            # Read the video
            cap = cv2.VideoCapture(video_path)

            # Create the directory to save frames
            os.makedirs('tmp', exist_ok=True)

            # Initialize frame counter
            frame_count = 0

            # Read and save frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                frame_path = os.path.join('tmp', f'image_{frame_count:05d}.jpg')
                cv2.imwrite(frame_path, frame)

            # Release the video capture
            cap.release()
            result = classify_video('tmp', input_file, class_names, model, opt)
            outputs.append(result)

            # Remove the directory tmp with all its contents
            shutil.rmtree('tmp')
        else:
            print('{} does not exist'.format(input_file))

    if os.path.exists('tmp'):
        shutil.rmtree('tmp')

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
