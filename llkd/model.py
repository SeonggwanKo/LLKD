import torch
import torch.nn as nn
import torch.nn.functional as F


class Mul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.matmul(x1,x2)

class Cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        return torch.cat([x2, x1], dim=1)


class Sum(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.add(x1,x2)




# ! Two downsampling
class student(nn.Module):

	def __init__(self):
		super(student, self).__init__()

		self.cat = Cat()
		self.relu = nn.ReLU(inplace=True)
		self.sigm = nn.Sigmoid()
		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.bn = nn.GroupNorm(4,32)
		number_f = 4


		self.conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.conv7 = nn.Conv2d(number_f*2,3,3,1,1,bias=True) 

		self.upsample0 = nn.ConvTranspose2d(number_f, number_f, kernel_size=4, stride=2,padding=1)


	def forward(self, x ):
		
		# Down Sampling Block
		x1 = self.relu(self.conv1(x))
		x1_down = self.maxpool(x1)
		x2 = (self.relu(self.conv2(x1_down)))
		x2_down = self.maxpool(x2)
		x3 = self.relu(self.conv3(x2_down))

		x4 = self.relu(self.conv4(x3))

		# Up Sampling Block 
		x5 = self.relu(self.conv5(self.cat(x3,x4)))
		x5_up = self.upsample0(x5)
		x6 = (self.relu(self.conv6(self.cat(x2,x5_up))))
		x6_up = self.upsample0(x6)
		x7 = self.relu(self.conv7(self.cat(x1,x6_up)))

		layers = [x1_down,x2_down,x5,x6]
		return x7,layers



class teacher(nn.Module):

	def __init__(self):
		super(teacher, self).__init__()

		self.cat = Cat()
		self.mul = Mul()
		self.sum = Sum()
		self.relu = nn.ReLU(inplace=True)
		self.sigm = nn.Sigmoid()
		self.bn = nn.GroupNorm(4,32)
		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		number_f = 32

		# Down Sampling Block
		self.conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.conv2 = nn.Conv2d(number_f,number_f*2,3,1,1,bias=True) 
		self.conv3 = nn.Conv2d(number_f*2,number_f*4,3,1,1,bias=True) 
		self.conv4 = nn.Conv2d(number_f*4,number_f*8,3,1,1,bias=True) 
		self.conv4_1 = nn.Conv2d(number_f*8,number_f*16,3,1,1,bias=True) 


		# Global Context Block
		self.g_conv1= nn.Conv2d(number_f*16, number_f*4,1,1,0,bias=True)
		self.g_conv2= nn.Conv2d(number_f*16, number_f*4,1,1,0,bias=True)
		self.g_conv3= nn.Conv2d(number_f*16, number_f*4,1,1,0,bias=True)
		self.g_conv4= nn.Conv2d(number_f*4, number_f*16,1,1,0,bias=True)
		self.g_conv5= nn.Conv2d(number_f*16, number_f*8,3,1,1,bias=True)

		# Up Sampling Block 
		self.upsample0 = nn.ConvTranspose2d(number_f*8, number_f*8, kernel_size=4, stride=2,padding=1)
		self.conv4_2 = nn.Conv2d(number_f*8 + number_f*8,number_f*4,3,1,1,bias=True) 
		self.upsample1 = nn.ConvTranspose2d(number_f*4, number_f*4, kernel_size=4, stride=2,padding=1)
		self.conv5 = nn.Conv2d(number_f*4 + number_f*4,number_f*2,3,1,1,bias=True) 
		self.upsample2 = nn.ConvTranspose2d(number_f*2, number_f*2, kernel_size=4, stride=2,padding=1)
		self.conv6 = nn.Conv2d(number_f*2 + number_f*2,number_f,3,1,1,bias=True) 
		self.upsample3 = nn.ConvTranspose2d(number_f, number_f, kernel_size=4, stride=2,padding=1)
		self.conv7 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.conv8 = nn.Conv2d(number_f,3,3,1,1,bias=True) 



	def forward(self, x):

		x1 = self.relu(self.conv1(x))
		p1 = self.maxpool(x1)
		x2 = self.relu(self.conv2(p1))
		p2 = self.maxpool(x2)
		x3 = self.relu(self.conv3(p2))
		p3 = self.maxpool(x3)
		x4 = self.relu(self.conv4(p3))
		p4 = self.maxpool(x4)
		x4_1 = self.relu(self.conv4_1(p4))

		att1 = self.g_conv1(x4_1)
		b,c,h,w = att1.shape
		att1 = att1.view(b,c,-1)
		att1 = att1.transpose(2,1)
		att2 = self.g_conv2(x4_1).view(b,c,-1)
		att3 = self.g_conv3(x4_1).view(b,c,-1)
		att3 = att3.transpose(2,1)
		att_map = F.softmax(torch.matmul(att1,att2), dim=1)
		applied_att_map = torch.matmul(att_map,att3).permute(0, 2, 1)
		applied_att_map = applied_att_map.view(b,c,h,w)
		applied_att_map = self.g_conv4(applied_att_map)
		z = self.sum(applied_att_map, x4_1)
		zz = self.g_conv5(z)
		
		m,s = torch.mean(att_map), torch.std(att_map)
		att_map_norm = (att_map - m)/s

		x4_2 = self.relu(self.conv4_2(self.cat(self.upsample0(zz),x4)))
		x5 = self.relu(self.conv5(self.cat(self.upsample1(x4_2),x3)))
		x6 = self.relu(self.conv6(self.cat(self.upsample2(x5),x2)))
		x7 = self.relu(self.conv7(self.cat(self.upsample3(x6),x1)))
		x_r = self.relu(self.conv8(x7))

		layers = [p1,p2,x5, x6]
		return x_r, att_map, layers
