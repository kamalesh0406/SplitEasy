import torch
import torch.nn as nn
import torchvision.models as models

class SplitA(nn.Module):
	def __init__(self):
		super(SplitA, self).__init__()
		layers = [module for module in models.vgg19(pretrained=False).features.modules() if type(module) != nn.Sequential]
		self.model = nn.Sequential(*layers[0:3])
	def forward(self, x):
		output = self.model(x)
		return output

class SplitB(nn.Module):
	def __init__(self):
		super(SplitB, self).__init__()

		conv_layers = [module for module in models.vgg19(pretrained=False).features.modules() if type(module) != nn.Sequential][3:]
		avgpool = [module for module in models.vgg19(pretrained=False).avgpool.modules() if type(module) != nn.Sequential]
		self.conv_model = nn.Sequential(*conv_layers,
			*avgpool)

		classifier = [module for module in models.vgg19(pretrained=False).classifier.modules() if type(module) != nn.Sequential][:-1]
		self.fc_model = nn.Sequential(*classifier)
	def forward(self, x):
		output = self.conv_model(x)
		output = output.view(-1, 25088)
		output = self.fc_model(output)
		return output

class SplitC(nn.Module):
	def __init__(self):
		super(SplitC, self).__init__()
		layers = [module for module in models.vgg19(pretrained=False).modules() if type(module) != nn.Sequential]
		self.model = nn.Sequential(*layers[-1:])
	def forward(self, x):
		output = self.model(x)
		return output

print(SplitA())
