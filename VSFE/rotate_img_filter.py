from glob import glob
from shutil import copyfile
import cv2
import pickle
import dlib
import os
from tqdm import tqdm

face_detector = dlib.get_frontal_face_detector()


def landmark(img):
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) > 0:
        return True
    return False


base_path = glob('./vsfe_dataset/rotate_frames/*')
for base in tqdm(base_path):
    filter_path = base.replace('rotate_frames', 'rotate_filtered_frames')
    os.makedirs(filter_path, exist_ok=True)
    img_path = glob(base + '/*')
    for img in tqdm(img_path):
        fig = cv2.imread(img)
        if landmark(fig):
            copyfile(img, img.replace('rotate_frames', 'rotate_filtered_frames'))
