import os
import ast
from flask import Flask , jsonify, request, Response
from flask_socketio import SocketIO, emit
from sys import getsizeof
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import time
import orjson
import mgzip
import inception
import inception_resnet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class ResNetSplitB(nn.Module):
    def __init__(self):
        super(SplitB, self).__init__()
        resnet_model = models.resnet50(pretrained=False)
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
        output = output.view(-1, 2048)
        return output

class VGGSplitB(nn.Module):
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

class InceptionV3SplitB(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception.inceptionv3()
    def forward(self, x):
        output = self.model(x)
        output = output.view(-1, 2048)
        return output

class InceptionResNetSplitB(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_resnet.inceptionresnetv2(pretrained=False, num_classes=1000)
    def forward(self, x):
        output = self.model(x)
        output = output.view(-1, 1536)
        return output

class DenseNetSplitB(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.densenet121().features
        self.features.conv0 = Identity()
        self.features.norm0 = Identity()
        self.features.relu0 = Identity()
        self.features.pool0 = Identity()
    def forward(self, x):
        output = self.features(x)
        output = nn.functional.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(-1, 1024)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=180, ping_interval=10)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
output_dict_B = []
output_dict_gradB = []
input_to_B = []
output_from_B = []
state = "normal"

@app.route('/outputToB', methods=['GET','POST'])
def forward_pass():
    global output_from_B
    global input_to_B
    global output_dict_B
    global state
    if request.method == "GET":
        state="normal"
        data = mgzip.compress(orjson.dumps(output_dict_B), 4)
        response = app.response_class(response=data, status=200, mimetype='application/json')
        response.headers['Content-Encoding'] = 'gzip'
        return response

    elif request.method == "POST":
        start = time.time()
        str_data = request.data
        data = orjson.loads(str_data)['inputs']
        parsing_end = time.time()
        print("Parsing Time in B", parsing_end-start)
        input_to_B = torch.Tensor(data).to(torch.float32)
        input_to_B = input_to_B.permute(0, 3, 1, 2)
        input_to_B.requires_grad = True
        output_from_B = model(input_to_B.to(device))
        server_B = output_from_B.detach().cpu()
        server_B = server_B.numpy().tolist()
        end = time.time()
        print("Time taken for forward pass in B", end-parsing_end)
        output_dict_B = {"inputs": server_B}
        state = "output_to_B"
        socketio.emit("state", state)
        return Response(status=201)

@app.route('/gradientsFromB', methods=['GET','POST'])
def back_propagation():
    global output_dict_gradB
    global state
    global output_from_B
    global input_to_B

    if request.method == "GET":
        start = time.time()
        state="normal"
        data = mgzip.compress(orjson.dumps(output_dict_gradB), 4)
        end = time.time()
        response = app.response_class(response=data, status=200, mimetype='application/json')
        response.headers['Content-Encoding'] = 'gzip'
        return response

    elif request.method == "POST":
        start = time.time()
        str_gradients = request.data
        data = orjson.loads(str_gradients)['gradients']
        parsing_end = time.time()
        print("Parsing Time for backward pass", parsing_end-start)
        gradient = torch.Tensor(data).to(torch.float32)
        print(gradient.size())
        targetb = output_from_B.detach().cpu() - gradient
        lossb = torch.nn.MSELoss(reduction='sum')(output_from_B , targetb.to(device))
        optimizer.zero_grad()
        lossb.backward()
        optimizer.step()
        output_grad = input_to_B.grad.detach().cpu().permute(0, 2, 3, 1)
        # print("Bytes of gradient from B", getsizeof(output_grad.storage()))
        output_dict_gradB = {"gradients": output_grad.numpy().tolist()}
        end = time.time()
        print("Time for backward pass in B", end-parsing_end)
        state = "gradients_from_B"
        socketio.emit("state", state)
        return Response(status=201)

port = int(os.environ.get('PORT', 8080))
if __name__=="__main__":
    args = parser.parse_args()
    if args.model_name == "resnet":
        model = ResNetSplitB()
    elif args.model_name == "inception":
        model = InceptionV3SplitB()
    elif args.model_name == "inception_resnet":
        model = InceptionResNetSplitB()
    elif args.model_name == "densenet": 
        model = DenseNetSplitB()
    elif args.model_name == "vgg":
        model = VGGSplitB()

    model.to(torch.float32)
    model.to(device)
    socketio.run(app, host='0.0.0.0', debug=True, port=port)
