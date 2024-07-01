import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, Subset
import random
from itertools import chain

import transform
from userutil import plot
from userutil import condition_function
import utils

class PA_Dataset(torch.utils.data.Dataset):
	#torch_geometric.data.Dataset:
	def __init__(self, args, type , mode):
		self.args = args
		self.mode = mode
		self.lis_dir = '/project/common/LGD/spice_data/output'
		self.raw_dir = '/project/common/LGD/spice_data/raw'
		os.makedirs(self.lis_dir, exist_ok=True)
		os.makedirs(self.raw_dir, exist_ok=True)
		
		cases = args['cases']
		data_files = []
		
		for c in cases:
			number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio = c
			# pixel_name = f'{number_scanline_pixel}_{number_dataline_pixel}_{res:.1f}_{cap:.3f}_{tw_s}_{VDH:.1f}_{load_ratio}'
			pixel_name = utils.getCircuitName('Pixel3T1C', (number_scanline_pixel, step_x, number_dataline_pixel, step_y, f'{res:.1f}', cap, f'TH*{tw_s}', VDH, load_ratio))
			if args['verbose']:
				print(pixel_name)
			param_dir = f'/project/common/LGD/spice_data/processed/{pixel_name}/param'
			datas_dir = f'/project/common/LGD/spice_data/processed/{pixel_name}/datas'			
			os.makedirs(param_dir, exist_ok=True)
			os.makedirs(datas_dir, exist_ok=True)
			data_files.append(self.construct_datasets(args, c, pixel_name, param_dir, datas_dir))
		
		flat_data_files = list(chain.from_iterable(data_files))
		# print(flat_data_files)
		self.datas = []
		for raw in flat_data_files:
			# print(raw)
			name = raw.split('/')[-1]
			base_dir = '/'.join(raw.split('/')[:-2])
			param_dir = os.path.join(base_dir, 'param')
			datas_dir = os.path.join(base_dir, 'datas')
			param = torch.load(os.path.join(param_dir, name))
			data = torch.load(os.path.join(datas_dir, name))
			if type == -1:
				for index in range(args['input_size']):
					d = data[index, int(args['start']):int(args['end']):int(1/args['resolution'])]
					# print(d.shape)
					t = torch.tensor([index], dtype=torch.float32)
					self.datas.append((d[0:args['lstm_window_size']], d, torch.cat([param, t])))
			
			else:
				d = data[type]
				d = d[int(args['start']):int(args['end']):int(1/args['resolution'])]
				# print(d.shape)
				t = torch.tensor([type], dtype=torch.float32)
				self.datas.append((d[0:args['lstm_window_size']], d, torch.cat([param, t])))

	def __len__(self):
		return len(self.datas)

	def __getitem__(self, idx):
		return self.datas[idx]
	
	def construct_datasets(self, args, c, pixel_name, param_dir, datas_dir):
		number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio = c
		# name = utils.getCircuitName('Pixel3T1C', (number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio))
		lis_file = os.path.join(self.lis_dir, f'{pixel_name}.lis')
		raw_file = os.path.join(self.raw_dir, f'raw{pixel_name}.raw')
		label_file = os.path.join(self.raw_dir, f'label{pixel_name}.raw')
		# print(lis_file)
		if not os.path.exists(raw_file):
			if args['verbose']:
				print("RAWFILE NOT Exists")
			
			if not os.path.exists(lis_file):
				if args['verbose']:
					print("SPICE File NOT Exists")		
					# raise FileNotFoundError
					return
			else:			
				transform.lis_file_parse(lis_file, raw_file, label_file)

		rawdata = torch.load(raw_file)
		label_name = ['time']
		with open(label_file, 'r') as file:
			label_name += file.readline().strip().split()
						
		if args['verbose']:
			print(f"Constructing {pixel_name} dataset...")
					
		min_max_dir = os.path.join(self.raw_dir, 'max_min')
		os.makedirs(min_max_dir, exist_ok=True)
		max_file, min_file = os.path.join(min_max_dir, f'max.np'), os.path.join(min_max_dir, f'min.np')
		v_max = np.array([-1e10, -1e10, -1e10, -1e10, -1e10])
		v_min = np.array([1e10, 1e10, 1e10, 1e10, 1e10])
		if os.path.exists(max_file) and os.path.exists(min_file):
			v_max = np.loadtxt(os.path.join(min_max_dir, f'max.np'), dtype=float)
			v_min = np.loadtxt(os.path.join(min_max_dir, f'min.np'), dtype=float)
			
		querys = ['scan', 'data', 'drg', 'drs', 'xel']
		v_max_, v_min_ = [], []				
		for index, query in enumerate(querys):	
			selected_rows = [row for i, row in enumerate(rawdata) if condition_function(label_name[i], query)]
			if selected_rows == []:				
				v_max_.append(v_max[index])
				v_min_.append(v_min[index])
				continue
			rawdata_ = torch.stack(selected_rows, dim=0)
			tp_v_max, tp_v_min = torch.max(rawdata_).item(), torch.min(rawdata_).item()
			v_max_.append(tp_v_max if tp_v_max > v_max[index] else v_max[index])
			v_min_.append(tp_v_min if tp_v_min < v_min[index] else v_min[index])
			
		np.savetxt(os.path.join(min_max_dir, f'max.np'), np.array(v_max_), delimiter=' ', fmt='%s')
		np.savetxt(os.path.join(min_max_dir, f'min.np'), np.array(v_min_), delimiter=' ', fmt='%s')
		# if args['verbose']:
		# 	print('Max : ', v_max_, 'Min : ', v_min_)
		# Pixel wise split
		data_files = []
		data_querys = ['--', '---', 'drg', 'drs', 'xel']
		if args['verbose']:
			plot_data = []

		for i in range(1, number_scanline_pixel + 1, step_x):
			for j in [2] + list(range(step_y, number_dataline_pixel + 1, step_y)):
				name = f"pixel{i}_{j}.praw"
				target = f"x_{i}_{j}.xwhite"
				params = torch.tensor([i/2000.0, number_scanline_pixel/2000.0, j/2000.0, number_dataline_pixel/2000.0, float(res)/8, cap, float(tw_s)/2.0, float(VDH)/10.0, float(load_ratio)/10.0])
				torch.save(params, os.path.join(param_dir, name))   
				
				selected_rows = [row for index, row in enumerate(rawdata) if condition_function(label_name[index], target)]
				selected_labels = [label for label in label_name if condition_function(label, target)]   
				if selected_rows == []:
					continue
				
				rawdata_ = torch.stack(selected_rows, dim=0)
				datas = []
				for index1, query in enumerate(data_querys):
					selected_rows_ = [row for index, row in enumerate(rawdata_) if condition_function(selected_labels[index], query)]
					if selected_rows_ == []:
						continue
					
					rawdata__ = torch.stack(selected_rows_, dim=0)
					datas.append((rawdata__ - v_min[index1]) / (v_max[index1] - v_min[index1]))
				
				if args['verbose']:
					plot_data.append(datas)

				datas_ = torch.stack(datas).squeeze()				
				torch.save(datas_, os.path.join(datas_dir, name))
				data_files += [os.path.join(datas_dir, name)]
				# if args['verbose']:
				# 	print('Saving : ', name)

		if args['verbose']:
			print('Plotting : ', pixel_name)
			self.data_plot(pixel_name, plot_data)
		return data_files

	def data_plot(self, pixel_name, datas):
		folder = '../original_data/plot/'
		os.makedirs(folder, exist_ok=True)
		file_name = ['drg', 'drs', 'diode']
		for i in range(0, 3):
			# print('Plotting : ', file_name[i])
			plt.clf()			
			plt.ylim(0,1)
			for j in range(len(datas)):
				d = datas[j][i][0]
				d = d[int(self.args['start']):int(self.args['end']):int(1/self.args['resolution'])]
				x = range(len(d))
				plt.plot(x, d, 'b')

			# plt.legend()
			plt.savefig(folder + file_name[i]+'_'+ pixel_name + '.png')

class MyCollator(object):
	def __init__(self, batch_size, mode):
		self.batch_size = batch_size
		self.mode = mode

	def __call__(self, samples):
		x_list, y_list, input_list = zip(*samples)
		diff = self.batch_size - len(x_list)
		x_list = torch.stack(x_list)
		y_list = torch.stack(y_list)
		input_list = torch.stack(input_list)
		if diff > 0:
			x_list = torch.cat([x_list, torch.zeros(diff, x_list.shape[1])])
			if self.mode == 'train':
				y_list = torch.cat([y_list, torch.zeros(diff)])
			else:
				y_list = torch.cat([y_list, torch.zeros(diff, y_list.shape[1])])
			input_list = torch.cat([input_list, torch.zeros(diff, input_list.shape[1])])

		return x_list, y_list, input_list	
	

		# padded_batch = [torch.nn.functional.pad(torch.tensor(item), (0, max_len - len(item)), value=0) for item in batch]
		# return torch.stack(padded_batch)
		# print(samples)
		# if diff > 0:
		# 	print(samples[0].shape, samples[2].shape)
		# 	for i in range(diff):
		# 		samples.append((torch.zeros(size=(samples[0][0].shape[0])), torch.tensor([0]), torch.zeros(size=(samples[0][2].shape[0]))))

		# return samples


def load_datasets(args, type = -1, mode = 'train'):	
	if mode == 'train':				
		dataset = PA_Dataset(args, type, 'train')
		# train_ratio = 0.8
		dataset_size = len(dataset)
		# train_size = int(train_ratio * dataset_size)
		# test_size = dataset_size - train_size
		batch_size = args['batch_size']


		train_size = int(dataset_size/batch_size) * batch_size
		test_size = dataset_size - train_size
		train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

		collate_fn = MyCollator(batch_size, mode)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers = 15, collate_fn = collate_fn)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers = 15, collate_fn = collate_fn)
		return train_loader, test_loader
	
	else:
		dataset = PA_Dataset(args, type, mode='plot')
		batch_size = args['batch_size']

		# all_indices = list(range(len(dataset)))
		# subset_dataset = Subset(dataset, all_indices[::len(dataset)//(3*batch_size)][:3*batch_size])
		collate_fn = MyCollator(batch_size, mode)
		plot_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers = 15, collate_fn = collate_fn)
		return plot_loader
