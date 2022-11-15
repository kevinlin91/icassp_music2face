import os
os.environ['LIB_DARKNET'] = './libdarknet.so'
import cv2
import dlib
import pyyolo
import librosa
import soundfile as sf
import warnings
import glob
import csv
import numpy as np

warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)


# detect model setting
detector = pyyolo.YOLO("./cfg/yolov4-custom.cfg",
                           "./cfg/new_weights/yolov4-custom_last.weights",
                           "./cfg/new_violin.data",
                           detection_threshold=0.3,)
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


ids = os.listdir('./violin_expression_youtube/video')
ids = [x.replace('.mp4', '') for x in ids]
# print(ids)

for video_name in ids:
    violin_flag = False
    face_flag = False

    # file setting
    cap = cv2.VideoCapture('./violin_expression_youtube/video/{}.mp4'.format(video_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    audio, sr = librosa.load('./violin_expression_youtube/audio/{}.wav'.format(video_name), sr=22050)

    # mkdir dataset folder
    os.makedirs('./vsfe_dataset/original_audio/{}'.format(video_name), exist_ok=True)
    os.makedirs('./vsfe_dataset/original_frames/{}'.format(video_name), exist_ok=True)
    os.makedirs('./vsfe_dataset/filtered_landmarks/{}'.format(video_name), exist_ok=True)
    os.makedirs('./vsfe_dataset/filtered_frames/{}'.format(video_name), exist_ok=True)
    os.makedirs('./vsfe_dataset/filtered_audio/{}'.format(video_name), exist_ok=True)

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print('frame error')
            break
        # deal with violin
        detections = detector.detect(frame, rgb=False)
        if len(detections) > 0:
            violin_flag = True

        # deal with face
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        output_landmark = list()
        if (len(faces)) > 0:
            face_flag = True
        cv2.imwrite('./vsfe_dataset/original_frames/{}/{}.jpg'.format(video_name, video_name + '_' + str(count).zfill(6)), frame)
        audio_start = count * (int(sr/fps))
        audio_end = audio_start + 882
        if audio_end > len(audio):
            length = audio_end - audio_start
            diff = 882 - length
            output_audio = np.concatenate([audio[audio_start:len(audio)], np.zeros(int(diff))])
        else:
            output_audio = audio[audio_start:audio_end]
        sf.write('./vsfe_dataset/original_audio/{}/{}.wav'.format(video_name, video_name + '_' + str(count).zfill(6)), output_audio, sr)
        # ===================
        if violin_flag and face_flag:
            cv2.imwrite('./vsfe_dataset/filtered_frames/{}/{}.jpg'.format(video_name, video_name + '_' + str(count).zfill(6)), frame)
            landmarks = predictor(image=gray, box=faces[0])
            box_x = faces[0].right() - faces[0].left()
            box_y = faces[0].bottom() - faces[0].top()
            center_x = faces[0].left() + (box_x/2)
            center_y = faces[0].top() + (box_y/2)
            for i in range(68):
                output_landmark.append((landmarks.part(i).x - center_x) / box_x)
                output_landmark.append((landmarks.part(i).y - center_y) / box_y)
            with open('./vsfe_dataset/filtered_landmarks/{}/{}.csv'.format(video_name, video_name + '_' + str(count).zfill(6)), 'w') as csv_file:
                fw = csv.writer(csv_file)
                fw.writerow(output_landmark)
            sf.write('./vsfe_dataset/filtered_audio/{}/{}.wav'.format(video_name, video_name + '_' + str(count).zfill(6)), output_audio, sr)
        violin_flag = False
        face_flag = False
        if count % 100 == 0:
            print(f'{video_name} frames: {count} done')
        count += 1

    cap.release()
    cv2.destroyAllWindows()
