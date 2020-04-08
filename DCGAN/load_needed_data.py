# batch будет иметь:
#    истинное изображение, 
#    подписи к нему (5),
#    ложное изображение (соответствует неправильному описанию).
# Дискриминатор:
#    true изображение + описание ~ реальный пример
#    false изображение + описание ~ поддельный пример

import os
import torch
import numpy as np
from PIL import Image # PIL - это библиотека изображений Python, которая предоставляет
                      # интерпретатору Python возможности редактирования изображений.
from tqdm import tqdm
from torch.autograd import Variable # Variable оборачивает тензор
import skipthoughts
from torch.utils.data import Dataset

class dataset_Text_to_image(Dataset): # рассмотрим на примере цветов
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_flower_dataset()

    def load_flower_dataset(self):
        # Возвращает: список имен файлов, 
        #             словарь (5 описаний на каждое изображение)

        print ("---  Loading images  ---")
        self.img_files = []
        for f in os.listdir(os.path.join(self.data_dir, 'flowers')):
            self.img_files.append(f)

        print ('Total number of images : {}'.format(len(self.img_files)))

        print ("---  Loading text captions  ---")
        self.img_captions = {}
        for class_dir in tqdm(os.listdir(os.path.join(self.data_dir, 'text_c10'))):
            if not 't7' in class_dir:
                for cap_file in class_dir:
                    if 'txt' in cap_file:
                        with open(cap_file) as f:
                            captions = f.read().split('\n')
                        img_file = cap_file[:11] + '.jpg'
                        # 5 описаний на каждое изображение
                        self.img_captions[img_file] = captions[:5]

        print ("---  Loading Skip-thought Model  ---")
        model = skipthoughts.load_model()
        self.encoded_captions = {}

        print ("---  Encoding of image captions STARTED  ---")
        for image_file in self.img_captions:
            self.encoded_captions[image_file] = skipthoughts.encode(model, self.img_captions[image_file])

        print ("---  Encoding of image captions DONE  ---")

    def read_image(self, image_file_name):
        image = Image.open(os.path.join(self.data_dir, 'flowers/' + image_file_name))
        # reshape (64, 64, 3)
        return image

    def get_false_img(self, index):
        false_image_id = np.random.randint(len(self.img_files))
        if false_image_id != index:
            return self.img_files[false_image_id]

        return self.get_false_img(index)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        sample = {}
        sample['true_images'] = torch.FloatTensor(self.read_image(self.img_files[index]))
        sample['false_images'] = torch.FloatTensor(self.read_image(self.get_false_img(index)))
        sample['true'] = torch.FloatTensor(self.encoded_captions[self.img_files[index]])

        return sample