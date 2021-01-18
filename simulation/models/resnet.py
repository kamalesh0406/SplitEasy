import torch
import torch.nn as nn
import torchvision.models as models

class SplitA(nn.Module):
	def __init__(self):
		super(SplitA, self).__init__()
		layers = [module for module in models.resnet18(pretrained=False).features.modules() if type(module) != nn.Sequential]
		self.model = nn.Sequential(*layers[0:3])
	def forward(self, x):
		output = self.model(x)
		return output

class SplitB(nn.Module):
	def __init__(self):
		super(SplitB, self).__init__()
		resnet_model = models.resnet18()
		layer1 = resnet_model.layer1
		layer2 = resnet_model.layer2
		layer3 = resnet_model.layer3
		layer4 = resnet_model.layer4
		avgpool = resnet_model.avgpool

		self.conv = nn.Sequential(
			layer1,
			layer2,
			layer3,
			layer4,
			avgpool)
	def forward(self, x):
		output = self.conv(x)
		output = output.view(-1, 512)
		return output

class SplitC(nn.Module):
	def __init__(self):
		super(SplitC, self).__init__()
		layers = [module for module in models.resnet18(pretrained=False).modules() if type(module) != nn.Sequential]
		self.model = nn.Sequential(*layers[-1:])
	def forward(self, x):
		output = self.model(x)
		return output

print(SplitA())
