# Json file format:  [{"video": "name", "clips": [{"segment": [i, f], "importance": 1mportance, "features": [512 feature]} * number of segments]} * number of videos]

import json

with open('output.json', 'r') as f:
    input_data = json.load(f)

preprocessed_data = []
for item in input_data:
    video_name = item["video"]
    clips = item["clips"]
    for clip in clips:
        segment = clip["segment"]
        importance = clip["importance"]
        features = clip["features"]
        preprocessed_data.append((video_name, segment, importance, features))
        
