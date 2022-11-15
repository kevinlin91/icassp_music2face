import torch
from tqdm import tqdm
import numpy as np
import os
import librosa
import dlib
import cv2
import warnings
warnings.filterwarnings("ignore")
import pickle

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def load_pickle():
    with open('./VSFE/sequence_list_range16.pkl', 'rb')as f:
        return pickle.load(f)


def landmark(img):
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    landmarks = face_predictor(image=gray, box=faces[0])
    output_landmarks = []
    box_x = faces[0].right() - faces[0].left()
    box_y = faces[0].bottom() - faces[0].top()
    center_x = faces[0].left()
    center_y = faces[0].top()
    for point in range(0, landmarks.num_parts):
        x = landmarks.part(point).x
        y = landmarks.part(point).y
        output_landmarks.append((x - center_x) / box_x)
        output_landmarks.append((y - center_y) / box_y)
    return torch.Tensor(output_landmarks)


def feature_extraction(root_dir):
    sequence_list = load_pickle()
    sequence_train_list = list()
    sequence_val_list = list()
    for sequence in sequence_list:
        if 'QmzavUR0Lfs' in sequence[0] or 'YywIPhFiZ94' in sequence[0]:
            sequence_val_list.append(sequence)
        else:
            sequence_train_list.append(sequence)

    os.makedirs(f'./violin_feature_extraction_scale01_train/audio', exist_ok=True)
    os.makedirs(f'./violin_feature_extraction_scale01_train/landmark', exist_ok=True)
    os.makedirs(f'./violin_feature_extraction_scale01_val/audio', exist_ok=True)
    os.makedirs(f'./violin_feature_extraction_scale01_val/landmark', exist_ok=True)


    audio_dir = os.path.join(root_dir, 'filtered_audio')
    frame_dir = os.path.join(root_dir, 'rotate_filtered_frames')

    for index, sequence in tqdm(enumerate(sequence_train_list)):
        audio_sequence = list()
        for s in sequence:
            audio_path = os.path.join(audio_dir, s.replace('yaw_0.0_', '').replace('jpg', 'wav'))
            extract_feature = list()
            # audio
            y, sr = librosa.load(audio_path)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=16)
            extract_feature.append(chroma)
            tonal = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)
            extract_feature.append(tonal)
            audio = np.concatenate((chroma, tonal), axis=0)
            audio_features = torch.from_numpy(audio).unsqueeze(0).float()
            audio_sequence.append(audio_features)

        # landmark
        frame_path = os.path.join(frame_dir, sequence[-1])

        output_landmark = landmark(cv2.imread(frame_path))
        output_audio = torch.stack(audio_sequence)

        torch.save(output_audio, f'./violin_feature_extraction_scale01_train/audio/{index}.pt')
        torch.save(output_landmark, f'./violin_feature_extraction_scale01_train/landmark/{index}.pt')
    
    print('train data done')
    print(output_landmark.size(), output_audio.size())

    for index, sequence in tqdm(enumerate(sequence_val_list)):
        audio_sequence = list()
        for s in sequence:
            audio_path = os.path.join(audio_dir, s.replace('yaw_0.0_', '').replace('jpg', 'wav'))
            extract_feature = list()
            # audio
            y, sr = librosa.load(audio_path)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=16)
            extract_feature.append(chroma)
            tonal = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)
            extract_feature.append(tonal)
            audio = np.concatenate((chroma, tonal), axis=0)
            audio_features = torch.from_numpy(audio).unsqueeze(0).float()
            audio_sequence.append(audio_features)

        # landmark
        frame_path = os.path.join(frame_dir, sequence[-1])

        output_landmark = landmark(cv2.imread(frame_path))
        output_audio = torch.stack(audio_sequence)

        torch.save(output_audio, f'./violin_feature_extraction_scale01_val/audio/{index}.pt')
        torch.save(output_landmark, f'./violin_feature_extraction_scale01_val/landmark/{index}.pt')

    print('val data done')
    print(output_landmark.size(), output_audio.size())


if __name__ == '__main__':
    root_dir = './VSFE/vsfe_dataset'
    feature_extraction(root_dir)
