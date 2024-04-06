import os
import glob

import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np


class Synth90kDataset(Dataset):   
    '''
    CHARS = ['이', '노', '시', '무', '니', '울', '차', '경', '광', '기', '배', '제', '머', '모', '고', '거', '오', '허', '너', '후', '조', '미', '구', '자', '원', '부', '가', '러', '산', '바', '누', '인', '히', '루', '다', '도', '커', '아', '두', '소', '서', '저', '수', '남', '파', '어', '마', '버', '천', '충', '투', '로', '북', '강', '우', '느', '지', '대', '보', '사', '주', '디', '라', '더', '전', '나', '하', '호', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    '''    
    CHARS = ['ঁ', 'ং', 'অ', 'ই', 'উ', 'এ', 'ও', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', '়', 'া', 'ি', 'ী', 'ু', 'ে', 'ো', 'ৌ', '্', '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯', 'য়']
    CHARS.sort()

    
    
    CHAR2LABEL = {}
    LABEL2CHAR = {}
    cnt = 0
    for c in CHARS:
       CHAR2LABEL[c] = cnt
       LABEL2CHAR[cnt] = c  
       cnt += 1   


    '''		
    # CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    '''

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        paths = []
        texts = []

        # Define the paths for images and labels
        images_dir = os.path.join(root_dir, 'images')
        labels_dir = os.path.join(root_dir, 'labels')

        # Get the list of image files (supporting various formats: PNG, JPG, JPEG)
        image_files = sorted(glob.glob(os.path.join(images_dir, '*.[PpJj][NnGg][PpEe]')))
        
        for image_path in image_files:
            # Get the corresponding label file path
            label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

            # Read the label from the label file
            with open(label_path, 'r') as label_file:
                label = label_file.read().strip()

            paths.append(image_path)
            texts.append(label)

        return paths, texts


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
