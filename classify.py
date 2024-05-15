import torch
from torch.autograd import Variable

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding
from Prepare_dataset import Prepare_dataset

data_driver = Prepare_dataset()

def classify_video(video_dir, video_name, class_names, model, opt):
    assert opt.mode in ['score', 'feature']

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)

    video_outputs = []
    video_segments = []
    for i, (inputs, segments) in enumerate(data_loader):
        with torch.no_grad():
            inputs = Variable(inputs)
            outputs = model(inputs)

        video_outputs.append(outputs.cpu().data)
        video_segments.append(segments)

    video_outputs = torch.cat(video_outputs)
    video_segments = torch.cat(video_segments)
    
    total_frames = video_segments[-1].tolist()[-1]
    annotations = data_driver.get_annotations(video_name)
    
    results = {
        'video': video_name,
        'clips': [],
        'total_frames': total_frames,
        'annotations_count': len(annotations)
    }

    _, max_indices = video_outputs.max(dim=1)
    
    for i in range(video_outputs.size(0)):
        clip_results = {
            'segment': video_segments[i].tolist(),
            'importance': annotations[i]
        }

        if opt.mode == 'score':
            clip_results['label'] = class_names[max_indices[i]]
            clip_results['scores'] = video_outputs[i].tolist()
        elif opt.mode == 'feature':
            clip_results['features'] = video_outputs[i].tolist()

        results['clips'].append(clip_results)

    return results
