import torch

def evaluate(modela, modelb, modelc, device, test_loader):
	correct = 0
	total = 0
	modela.eval()
	modelb.eval()
	modelc.eval()
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0].to(device)
			target = data[1].to(device)

			outputa = modela(inputs)
			outputb = modelb(outputa)
			outputs = modelc(outputb)

			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()

	return (100*correct/total)
