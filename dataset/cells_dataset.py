import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from dataset.transform import normalize, crop

class CellsDataset(Dataset):
    # modes can be train or val
    def __init__(self, root, crop_size = 700, shot = 1, mode="train"):
        self._root = root
        self._mode = mode
        self.shot = shot
        self.crop_size = crop_size
        
        # Setting up paths
        self._img_path = os.path.join(root, 'images')
        self._mask_path = os.path.join(root, 'masks')
        self._id_path = os.path.join(root, 'ids')

        # Opening each image id
        with open(os.path.join(self._id_path, '%s.txt' % self._mode), 'r') as f:
            self._ids = f.read().splitlines()
        print(self._ids)

    def __getitem__(self, item):
        # Opening the query image
        query_id = self._ids[item]
        query_img = Image.open(os.path.join(self._img_path, query_id + ".png")).convert('RGB')
        query_mask = Image.fromarray(np.array(Image.open(os.path.join(self._mask_path, query_id + ".png")))).convert("L")

        support_img_list = []
        support_mask_list = []
        
        # Opening support images
        for id in self._ids:
          if id != query_id:
            support_img = Image.open(os.path.join(self._img_path, query_id + ".png")).convert('RGB')
            support_mask = Image.fromarray(np.array(Image.open(os.path.join(self._mask_path, query_id + ".png")))).convert("L")
            support_img_list.append(support_img)
            support_mask_list.append(support_mask)
          if len(support_img_list) >= self.shot:
            break

        #query_img, query_mask = query_img.thumbnail((self.crop_size, self.crop_size)), query_mask.thumbnail((self.crop_size, self.crop_size))
        query_img, query_mask = self.__preprocess_query_img(query_img, query_mask)
        support_img_list, support_mask_list = self.__preprocess_support_imgs(support_img_list, support_mask_list)


        return support_img_list, support_mask_list, query_img, query_mask

    def __preprocess_query_img(self, query_img, query_mask):
            '''Crops and normalizes query image'''
            preproc_query_img, preproc_query_mask = crop(query_img, query_mask, self.crop_size)
            preproc_query_img, preproc_query_mask = normalize(preproc_query_img, preproc_query_mask)
            return preproc_query_img, preproc_query_mask
        
    def __preprocess_support_imgs(self,support_img_list:list, support_mask_list: list) -> tuple[list, list]:
        '''Crops and normalizes support image list'''
        preproc_support_img_list = support_img_list.copy()
        preproc_support_mask_list = support_mask_list.copy()
        
        for k in range(len(support_img_list)):
            #support_img_list[k], support_mask_list[k] = support_img_list[k].resize((self.crop_size, self.crop_size)), support_mask_list[k].resize((self.crop_size, self.crop_size))
            #support_img_list[k], support_mask_list[k] = support_img_list[k].thumbnail((self.crop_size, self.crop_size)), support_mask_list[k].thumbnail((self.crop_size, self.crop_size))
            preproc_support_img_list[k], preproc_support_mask_list[k] = crop(support_img_list[k], support_mask_list[k], self.crop_size)
            preproc_support_img_list[k], preproc_support_mask_list[k] = normalize(preproc_support_img_list[k], preproc_support_mask_list[k])
        return preproc_support_img_list, preproc_support_mask_list
    
    def __len__(self):
        return len(self._ids)