### dataset.py
### object of gcn dataset

import os
import numpy as np
import random
import torch
from userutil import plot
import transform
import time
import matplotlib.pyplot as plt


class TotalDataset(torch.utils.data.Dataset):
	#torch_geometric.data.Dataset:
	def __init__(self, args, description, raw_files):
		self.args = args
		self.description = description
		self.raw_files = raw_files

	def __len__(self):
		return len(self.raw_files)

	def __getitem__(self, idx):
		return self.raw_files[idx]
		

def split_sequences(sequence, n_steps):
	X, y = torch.empty(1, n_steps), torch.empty(1, 1)
	for i in range(len(sequence) - n_steps - 1):
		seq_x, seq_y = sequence[i:i + n_steps], sequence[i + n_steps]
		seq_x = seq_x.unsqueeze(dim=0)
		seq_y = seq_y.view(-1,1)
		X = torch.cat([X, seq_x], dim=0)
		y = torch.cat([y, seq_y], dim=0)

	X, y = X[1:], y[1:]

	return X.unsqueeze(-1), y.unsqueeze(-1)

def construct_datasets(args, c, lis_dir, raw_dir, param_dir, datas_dir):
	num_scan_pixels, num_data_pixels, res, cap, tw_s, VDH = c
	pixel_name = f'{num_scan_pixels}_{num_data_pixels}_{res}_{cap}_{tw_s}_{VDH}'
	
	lis_file = os.path.join(lis_dir, f'Array{pixel_name}.lis')
	raw_file = os.path.join(raw_dir, f'raw{pixel_name}.raw')
	if not os.path.exists(raw_file):
		print("RAWFILE NOT Exists")
		if not os.path.exists(lis_file):
			print("SPICE File NOT Exists")
		else:			
			print(transform.lis_file_parse(os.path.join(lis_dir, f'Array{pixel_name}.lis'), raw_file))
	
	print(f"Constructing {pixel_name} dataset...")
	rawdata = torch.load(raw_file)

	min_max_dir = os.path.join(raw_dir, 'max_min')
	os.makedirs(min_max_dir, exist_ok=True)

	max_file, min_file = os.path.join(min_max_dir, f'max{pixel_name}.np'), os.path.join(min_max_dir, f'min{pixel_name}.np')
	if not os.path.exists(max_file) or not os.path.exists(min_file):
		v_max, v_min = transform.find_maxmin(rawdata[1:, ])		
		np_v_max, np_v_min = v_max.numpy(), v_min.numpy()
		np.savetxt(os.path.join(min_max_dir, f'max{pixel_name}.np'), v_max, delimiter=' ', fmt='%s')
		np.savetxt(os.path.join(min_max_dir, f'min{pixel_name}.np'), v_min, delimiter=' ', fmt='%s')

	v_max, v_min = transform.find_maxmin(rawdata[1:, ])		
	np_v_max, np_v_min = v_max.numpy(), v_min.numpy()
	np.savetxt(os.path.join(min_max_dir, f'max{pixel_name}.np'), v_max, delimiter=' ', fmt='%s')
	np.savetxt(os.path.join(min_max_dir, f'min{pixel_name}.np'), v_min, delimiter=' ', fmt='%s')
	np_v_max = np.loadtxt(os.path.join(min_max_dir, f'max{pixel_name}.np'), dtype=float)
	np_v_min = np.loadtxt(os.path.join(min_max_dir, f'min{pixel_name}.np'), dtype=float)
	v_max, v_min = torch.from_numpy(np_v_max), torch.from_numpy(np_v_min)
	v_max, v_min = v_max[10:], v_min[10:]
	v_max, v_min = v_max.to(torch.float64).unsqueeze(1).repeat(1, rawdata.shape[1]), v_min.to(torch.float64).unsqueeze(1).repeat(1, rawdata.shape[1])
	
	colors = ['r','b','g', 'y', 'k']
	color_ = 0
	endpixel = num_data_pixels - 1 + num_data_pixels * (num_scan_pixels - 1)
	dataplotdir = '/home/projects/MLCASim/plot/inputs/'
	x = rawdata[0]
	
	plt.clf()
	png_file = os.path.join(dataplotdir, f'drg{pixel_name}.png')
	for i in [1, 2, 3, endpixel - 1, endpixel]:
		plt.plot(x, rawdata[11 + 22*i,], colors[color_])
		color_= (color_ + 1)%5
	plt.savefig(png_file)

	plt.clf()
	_png_file = os.path.join(dataplotdir, f'scan{pixel_name}.png')
	# for i in range(0, 50, 10):
	for i in [1, 2, 3, endpixel - 1, endpixel]:
		plt.plot(x, rawdata[1 + 22*i], 'r')
		plt.plot(x, rawdata[2 + 22*i], 'b')

	plt.savefig(_png_file)

	plt.clf()
	_png_file = os.path.join(dataplotdir, f'drs{pixel_name}.png')
	for i in [1, 2, 3, endpixel - 1, endpixel]:
		plt.plot(x, rawdata[12 + 22*i,], colors[color_])
		color_= (color_ + 1)%5

	plt.savefig(_png_file)

	plt.clf()
	_png_file = os.path.join(dataplotdir, f'diode{pixel_name}.png')
	for i in [1, 2, 3, endpixel - 1, endpixel]:
		plt.plot(x, rawdata[13 + 22*i,], colors[color_])
		color_= (color_ + 1)%5

	plt.savefig(_png_file)

	plt.clf()
	_png_file = os.path.join(dataplotdir, f'data{pixel_name}.png')
	# for i in range(0, 50, 10):
	plt.plot(x, rawdata[3 + 22*1], 'r')
	plt.plot(x, rawdata[3 + 22*endpixel], 'k')

	plt.savefig(_png_file)

	data_files =[]
	for i in range(num_scan_pixels):
		for j in range(num_data_pixels):
			name = f"pixel{i}_{j}.praw"
			index = 1 + 22 * (j + num_data_pixels * i)
			# if not os.path.exists(os.path.join(param_dir, name)):
			params = torch.tensor([i, num_scan_pixels, j, num_data_pixels, float(res), cap, tw_s, VDH])
			torch.save(params, os.path.join(param_dir, name))
		
			# if not os.path.exists(os.path.join(datas_dir, name)):
			datas = (rawdata[index+10:index+22, :] - v_min) / (v_max - v_min)
			torch.save(datas, os.path.join(datas_dir, name))
			
			data_files += [os.path.join(datas_dir, name)]

	return data_files

def get_datas(args, c, mode = 'train'):
	num_scan_pixels, num_data_pixels, res, cap, tw_s, VDH = c
	pixel_name = f'{num_scan_pixels}_{num_data_pixels}_{res}_{cap}_{tw_s}_{VDH}'

	lis_dir = '/project/common/LGD/spice_data/output'
	raw_dir = '/project/common/LGD/spice_data/raw'
	param_dir = f'/project/common/LGD/spice_data/processed/{pixel_name}/param'
	datas_dir = f'/project/common/LGD/spice_data/processed/{pixel_name}/datas'

	os.makedirs(raw_dir, exist_ok=True)
	os.makedirs(param_dir, exist_ok=True)
	os.makedirs(datas_dir, exist_ok=True)

	data_files = construct_datasets(args, c, lis_dir, raw_dir, param_dir, datas_dir)

	datas = []
	for raw in data_files:
		name = raw.split('/')[-1]

		param = torch.load(os.path.join(param_dir, name))
		data = torch.load(os.path.join(datas_dir, name))	

		x, y = 0, 0
		for index in range(data.shape[0]):
			d = data[index, int(args['start']):int(args['end']):int(1/args['resolution'])]
			dx, dy = split_sequences(d, args['lstm_window_size'])
			if index == 0:
				x = torch.zeros_like(dx, dtype=torch.float64)
				y = torch.zeros_like(dy, dtype=torch.float64)
			x = torch.cat([x, dx], dim=2)
			y = torch.cat([y, dy], dim=2)
		
		# params = params.to(torch.float64)
		datas.append((x[:, :, 1:,], y[:, :, 1:,], param))

	return datas


def get_datasets(args, num_train, num_test, datas, mode = 'train'):
	all_indices = list(range(len(datas)))

	if mode == 'train':
		indices1 = random.sample(all_indices, num_train)
		remaining_indices = list(set(all_indices) - set(indices1))
		indices2 = random.sample(remaining_indices, num_test)

		train_dataset = TotalDataset(args = args, description='train', raw_files=[datas[i] for i in indices1])
		test_dataset  = TotalDataset(args = args, description='test' , raw_files=[datas[i] for i in indices2])
		return (train_dataset, test_dataset)
	
	else:
		indices1 = all_indices[::len(datas)//num_train][:num_train]
		plot_dataset  = TotalDataset(args = args, description='plot' , raw_files=[datas[i] for i in indices1])
		return plot_dataset
