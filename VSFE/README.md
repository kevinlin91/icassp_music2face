# Violin Soundtrack and Facial Expression dataset

### Preparing YOLOv4 model for violin detection
1. Clone the YOLOv4 model from https://github.com/AlexeyAB/darknet

2. Modify two arguments in darknet/Makefile:

```
OPENCV=1
LIBSO=1
```        

3. Complie (make) the darknet model to obtain "libdarknet.so"

4. Put libdarknet.so under VSFE folder

5. Download the pre-trained YOLOv4 model from https://drive.google.com/file/d/1fmx5PsuFPTk8nzX-WDqJxnWrqEaRAIWf/view?usp=sharing, put the files under VSFE/cfg folder

### Preparing face detection model
1. Install python package: DLib

2. Put shape_predictor_68_face_landmarks.dat (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) under VSFE folder

### Preparing face rotation model
1. Clone the Rotate-and-Render from https://github.com/Hangz-nju-cuhk/Rotate-and-Render

2. Put the repository under VSFE folder and follow the instructions on Rotated-and-Render to build the rotation model

3. Modify Rotate-and-Render/experiments/v100_test.sh

```
--gpu_ids 0
--nThreads 1
--yaw_poses 0
```

Modify Rotate-and-Render/test_multipose.py

```
line102: opt.gpu_ids = list(range(0, ngpus - opt.render_thread)) to opt.gpu_ids = [0]
line109: opt.gpu_ids[-1] to opt.gpu_ids[0]
line128: opt.gpu_ids[-1] to opt.gpu_ids[0]
```

Modify Rotate-and-Render/3ddfa/inference.py
    
```
line82: pts_2d_68 = preds[0][0] to pts_2d_68 = preds[0]
```
		
### Start building the dataset
1. Run violin_expression_ytdl.py to download the videos from YouTube
		python violin_expression_ytdl.py
	The results will be saved in VSFE/violin_expression_youtube
	
	Audio &raquo; VSFE/violin_expression_youtube/audio
    
	Video &raquo; VSFE/violin_expression_youtube/video
	
2. Run violin_expression_extract.py to extract/filter frames and audio for the YouTube videos
		python violin_expression_extract.py
	The results will be saved in VSFE/vsfe_dataset
	
	Original audio &raquo; VSFE/vsfe_dataset/original_audio
	Original frames &raquo; VSFE/vsfe_dataset/original_frames
	Extracted audio &raquo; VSFE/vsfe_dataset/filtered_audio
	Extracted frames &raquo; VSFE/vsfe_dataset/filtered_frames
	Extracted landmark &raquo; VSFE/vsfe_dataset/filtered_landmarks
	
3. Run run_rotated.sh to obtain the rotated face images
		run $sh run_rotate.sh
	The rotated face images will be saved in VSFE/vsfe_dataset/rotate_frames
	The filtered images will be saved in VSFE/vsfe_dataset/rotate_filtered_frames
	
4. Run sequence_filter.py to obtain the index of the sequence data
		python sequence_filter.py
	This python file will generate two outputs (sequence_list_more_than_17.pkl and sequence_list_range16.pkl)
- sequence_list_more_than_17.pkl: This file contains sequences which are longer than 17
- sequence_list_range16.pkl: Every sequence in this file is fix to 16. The seuquences here are extracted from the sequences in sequence_list_more_than_17.pkl