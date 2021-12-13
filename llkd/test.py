import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import Myloss


def lowlight(image_path):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	student = model.student().to(device)

	student.load_state_dict(torch.load('snapshots/student.pth'))

	pred, _ = student(data_lowlight)
	pred_path = image_path.replace('test','result')

	result_pred_path = pred_path
	if not os.path.exists(pred_path.replace('/'+pred_path.split("/")[-1],'')):
		os.makedirs(pred_path.replace('/'+pred_path.split("/")[-1],''))
	torchvision.utils.save_image(pred, result_pred_path)


if __name__ == '__main__':
	# test_images
	with torch.no_grad():
		filePath = 'dataset/test/'
		file_list = os.listdir(filePath)
		for idx in file_list :
			end_time = 0
			test_list = glob.glob(filePath+idx+"/*") 
			n_img = len(test_list)
			print('[*] Testing :', str(len(test_list)) + " " + (filePath+idx))
			for image in test_list:
				lowlight(image)
		print('[*] Done')



