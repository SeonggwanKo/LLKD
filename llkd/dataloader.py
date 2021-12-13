import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import copy

import matplotlib.pyplot as plt
from typing import Union, List, Any
from torchvision.utils import save_image

import time

random.seed(1143)


def populate_train_list(lowlight_images_path):

	image_list_lowlight = glob.glob(lowlight_images_path + "*")
	train_list = image_list_lowlight
	random.shuffle(train_list)

	return train_list

def change_name(l, lowlight_images_path, gt_images_path) :
	new = []
	for i in l :
		new.append(i.replace(lowlight_images_path, gt_images_path).replace('jpg','jpg'))

	return new

class lowlight_loader(data.Dataset):
	def __init__(self, lowlight_images_path, gt_images_path):

		self.train_list = populate_train_list(lowlight_images_path) 
		self.gt_list = change_name(self.train_list, lowlight_images_path, gt_images_path)
		self.size = 128
		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))

	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		gt_path = self.gt_list[index]

		# print('start')
		# start = time.time()
		data_lowlight = Image.open(data_lowlight_path)
		gt = Image.open(gt_path)
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		gt = gt.resize((self.size, self.size), Image.ANTIALIAS)

		# end_time = (time.time() - start)
		# print('test',end_time)
		gt = np.asarray(gt)/255.0
		gt = torch.from_numpy(gt).float()
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1), gt.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)


def randomCrop(size,image):
	height, width = image.shape[:2]
	new_height = size
	new_width = size

	lefttop_Y = np.random.randint(0, height - new_height)
	lefttop_X = np.random.randint(0, width - new_width)

	range_Y = np.arange(lefttop_Y, lefttop_Y + new_height, 1)[:, np.newaxis].astype(np.int32)
	range_X = np.arange(lefttop_X, lefttop_X + new_width, 1).astype(np.int32)

	return image[range_Y, range_X]


