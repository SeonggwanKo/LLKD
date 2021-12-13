import torch
import torch.optim
import os
import argparse
import dataloader
import model
import Myloss
from optimizer import AdaBelief

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


torch.cuda.manual_seed(1)
torch.manual_seed(1)

def train(config):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	student = model.student().to(device)
	teacher = model.teacher().to(device)
	if config.load_pretrain == True:
		teacher.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path, config.gt_images_path)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	# Loss
	L_exp = Myloss.L_exp(64,0.7)
	L_mse = Myloss.L_mse()
	L_mse2 = Myloss.L_mse()
	L_feat_ext = Myloss.L_feat_ext()
	percep = Myloss.VGGPerceptualLoss()


	writer = SummaryWriter()
	optimizer = AdaBelief(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	student.train()
	teacher.eval()

	status_bar = tqdm(total=len(train_loader)*config.num_epochs, desc="Train is Started.")
	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):
			
			optimizer.zero_grad()
			img_lowlight, gt= img_lowlight
			img_lowlight, gt = img_lowlight.cuda(), gt.cuda()
			enhanced_image,_, layers_teacher = teacher(img_lowlight)
			preds, layers_student = student(img_lowlight)	

			sim_stu = []
			sim_tch = []
			sim_stu.append([Myloss.spatial_similarity(layer) for layer in layers_student])
			sim_tch.append([Myloss.spatial_similarity(layer) for layer in layers_teacher])


			loss_feature = L_feat_ext(sim_stu[0], sim_tch[0])
			loss_mse2 = L_mse2(preds, gt)
			loss_mse = L_mse(preds,enhanced_image)
			loss_percep = percep(preds, gt)
			loss_exp = torch.mean(L_exp(preds))

			loss = loss_feature+ (0.5*loss_mse +0.5*loss_mse2) + 0.2*loss_exp + 0.5*loss_percep 

			writer.add_scalar("Loss/train",loss,iteration)


			loss.backward()
			status_bar.set_description(desc=f"Epoch: {epoch+1:2d}    Loss: {loss.item():.5f}")
			torch.nn.utils.clip_grad_norm(student.parameters(),config.grad_clip_norm)
			optimizer.step()
			status_bar.update()
		if ((epoch+1) % config.snapshot_iter) == 0:
			torch.save(student.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="dataset/train/x/")
	parser.add_argument('--gt_images_path', type=str, default="dataset/train/y/")

	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--display_iter', type=int, default=100)
	parser.add_argument('--snapshot_iter', type=int, default=5)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= True)
	parser.add_argument('--pretrain_dir', type=str, default= "teacher.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	train(config)


