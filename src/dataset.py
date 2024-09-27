##################################################################
## This file contains the dataset class and collator for 		##
## Pixel Array (PA)   											##
## PA_Dataset : Dataset class for PA							##
## 				- __init__ : Constructor						##
## 				- __len__ : Return the length of the dataset	##
## 				- __getitem__ : Return the item of the dataset	##
## 				- parse_LisFile : Parse SPICE output (.lis) to	##	
## 									raw data and label			##
## 				- construct_config_datasets : Construct the		##
## 									dataset based on the		##
## 									configurations				##
## 				- collect_datasets : Collect the datasets		##
## PA_Collator : Collator class for PA							##
## load_datasets : Load the datasets based on the arguments		##
##################################################################
## Author: Jaeseung Lee                                         ##
## Date: September 2024                                         ##
## Affiliation: POSTECH CSDL, South Korea                       ##
##################################################################

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from itertools import chain

import utils


class PA_Dataset(torch.utils.data.Dataset):
	def __init__(self, args, type , normalization = 'linear', mode = 'train'):
		self.args = args
		self.mode = mode
		self.lis_dir = '/project/common/LGD/spice_data/output'
		self.raw_dir = '/project/common/LGD/spice_data/raw'
		self.process_dir = '/project/common/LGD/spice_data/processed'
		os.makedirs(self.lis_dir, exist_ok=True)
		os.makedirs(self.raw_dir, exist_ok=True)
		
		self.normalization = utils.linear_normalization if normalization == 'linear' else None
		self.datas = []
		
		## Data Construction
		data_files = self.collect_datasets(args)
		## Load the data
		for raw in data_files:
			name = raw.split('/')[-1]
			base_dir = '/'.join(raw.split('/')[:-2])
			param_dir = os.path.join(base_dir, 'param')
			datas_dir = os.path.join(base_dir, 'datas')
			param = torch.load(os.path.join(param_dir, name))
			data = torch.load(os.path.join(datas_dir, name))
			start, end, hop = int(args['start']), int(args['end']), int(1/args['resolution'])
			
			## All type. type = 0 : drg, type = 1 : drs, type = 2 : xel
			if type == -1:
				for index in range(args['input_size']):
					d = data[index, start:end:hop]
					# print(d.shape)
					t = torch.tensor([index], dtype=torch.float32)
					self.datas.append((d[0:args['window_size']], d, torch.cat([param, t])))
			
			else:
				d = data[type]
				d = d[start:end:hop]
				t = torch.tensor([type], dtype=torch.float32)
				self.datas.append((d[0:args['window_size']], d, torch.cat([param, t])))

	def __len__(self):
		return len(self.datas)

	def __getitem__(self, idx):
		return self.datas[idx]
	
	## Parse the SPICE output (.lis) to raw data and label
	def parse_LisFile(self, lis_file, raw_file, label_file):
		labels = []
		voltage_data = torch.empty([1,1])
		with open(lis_file, 'r') as file:
			lines = file.readlines()
			start_index = -1
			label_line_indices = []
			
			## Check the line where the transient analysis starts
			for index, line in enumerate(lines):
				if ' transient analysis' in line:
					start_index = index + 1
				if start_index != -1:
					if 'x' == line[0]:
						label_line_indices.append(index+3)

			storeFlag = 0
			prev = 0
			prec = 0
			## Extract the labels and voltage data
			for index in label_line_indices:
				labels.extend(lines[index].strip().split())
				label_num = len(lines[index].strip().split())
				volt = torch.empty([label_num,1])
				time = torch.empty([1])
				for line in lines[index+1:]:
					if line.strip() == '': 
						continue
					## Check the end of the data
					if line.strip() == 'y':
						if storeFlag == 0:
								voltage_data = time[1:].unsqueeze(0)
								storeFlag = 1
						voltage_data = torch.cat([voltage_data, volt[:, 1:]], dim=0)                    
						break

					values = line.strip().split()
					voltages = torch.tensor([float(value) for value in values[1:]]).unsqueeze(1)
					if storeFlag == 0:
							## Exception for same timestep (SPICE simulation precision problem)
							if prev == float(values[0]):
									prec = prec + 2e-10
									if prec >= 1e-9:
											prec = 0

							t = torch.tensor([float(values[0]) + prec])
							time = torch.cat([time, t], dim=0)
							prev = float(values[0])
							
					volt = torch.cat([volt, voltages], dim=1)

		## Save the labels
		with open(label_file, 'w') as file:
			file.write(' '.join(labels))
			
		## Save the raw voltage data
		torch.save(voltage_data, raw_file)
		return voltage_data.shape
    
	## Convert to the dataset based on the configurations
	def construct_config_datasets(self, args, config, pixel_name, param_dir, datas_dir):
		number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio = config
		lis_file = os.path.join(self.lis_dir, f'{pixel_name}.lis')
		raw_file = os.path.join(self.raw_dir, f'raw{pixel_name}.raw')
		label_file = os.path.join(self.raw_dir, f'label{pixel_name}.raw')
		
		if not os.path.exists(raw_file):
			if args['verbose']:
				print("RAWFILE NOT Exists")
			
			if not os.path.exists(lis_file):
				if args['verbose']:
					print("SPICE File NOT Exists")		
					# raise FileNotFoundError
					return
			else:			
				self.parse_LisFile(lis_file, raw_file, label_file)

		rawdata = torch.load(raw_file)
		label_name = ['time']

		with open(label_file, 'r') as file:
			label_name += file.readline().strip().split()
						
		if args['verbose']:
			print(f"Constructing {pixel_name} dataset...")
					
		min_max_dir = os.path.join(self.raw_dir, 'max_min')
		os.makedirs(min_max_dir, exist_ok=True)
		v_max, v_min = utils.load_maxmin(min_max_dir)
			
		querys = ['scan', 'data', 'drg', 'drs', 'xel']
		## Update the max and min values for normalization
		v_max_, v_min_ = [], []				
		for index, query in enumerate(querys):	
			selected_rows = [row for i, row in enumerate(rawdata) if utils.condition_function(label_name[i], query)]
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

		## Construct the torch.dataset
		data_files = []
		data_querys = ['--', '---', 'drg', 'drs', 'xel']
		if args['verbose']:
			plot_data = []

		for i in range(1, number_scanline_pixel + 1, step_x):
			for j in [2] + list(range(step_y, number_dataline_pixel + 1, step_y)):
				name = f"pixel{i}_{j}.praw"
				target = f"x_{i}_{j}.xwhite"
				params = utils.convert_param_to_tensor(i, number_scanline_pixel, j, number_dataline_pixel, res, cap, tw_s, VDH, load_ratio)
				torch.save(params, os.path.join(param_dir, name))
				
				selected_rows = [row for index, row in enumerate(rawdata) if utils.condition_function(label_name[index], target)]
				selected_labels = [label for label in label_name if utils.condition_function(label, target)]   
				if selected_rows == []:
					continue
				
				rawdata_ = torch.stack(selected_rows, dim=0)
				datas = []
				for index1, query in enumerate(data_querys):
					selected_rows_ = [row for index, row in enumerate(rawdata_) if utils.condition_function(selected_labels[index], query)]
					if selected_rows_ == []:
						continue
					
					rawdata__ = torch.stack(selected_rows_, dim=0)
					datas.append(self.normalization(rawdata__, v_max[index1], v_min[index1]))
				
				if args['verbose']:
					plot_data.append(datas)

				datas_ = torch.stack(datas).squeeze()				
				torch.save(datas_, os.path.join(datas_dir, name))
				data_files += [os.path.join(datas_dir, name)]

		if args['verbose']:
			print('Plotting : ', pixel_name)
			self.data_plot(pixel_name, plot_data)
		return data_files
	
	## Collect the datasets	for batch processing
	def collect_datasets(self, args):
		data_files = []
		cases = args['test_cases'] if self.mode == 'inference' else args['cases']

		for c in cases:
			number_scanline_pixel, step_x, number_dataline_pixel, step_y, res, cap, tw_s, VDH, load_ratio = c
			pixel_name = utils.getCircuitName('Pixel3T1C', (number_scanline_pixel, step_x, number_dataline_pixel, step_y, f'{res:.1f}', cap, f'TH*{tw_s}', VDH, load_ratio))
			if args['verbose']:
				print(pixel_name)
				
			param_dir = os.path.join(self.process_dir, f'{pixel_name}/param')
			datas_dir = os.path.join(self.process_dir, f'{pixel_name}/datas')	
			os.makedirs(param_dir, exist_ok=True)
			os.makedirs(datas_dir, exist_ok=True)
			
			data_files.append(self.construct_config_datasets(args, c, pixel_name, param_dir, datas_dir))
		return list(chain.from_iterable(data_files))

	## Plot the data for debugging
	def data_plot(self, pixel_name, datas):
		folder = '../original_data/plot/'
		os.makedirs(folder, exist_ok=True)
		file_name = ['drg', 'drs', 'diode']
		for i in range(0, 3):
			plt.clf()			
			plt.ylim(0,1)
			for j in range(len(datas)):
				d = datas[j][i][0]
				d = d[int(self.args['start']):int(self.args['end']):int(1/self.args['resolution'])]
				x = range(len(d))
				plt.plot(x, d, 'b')

			plt.savefig(folder + file_name[i]+'_'+ pixel_name + '.png')

## Collator for the PA dataset
class PA_Collator(object):
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

## Load the datasets based on the arguments
def load_datasets(args, type = -1, mode = 'train'):	
	if mode == 'train':				
		dataset = PA_Dataset(args, type, normalization=args['normalization'], mode = mode)
		dataset_size = len(dataset)
		batch_size = args['batch_size']
		train_size = int(dataset_size/batch_size) * batch_size
		test_size = dataset_size - train_size
		train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

		collate_fn = PA_Collator(batch_size, mode)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers = 15, collate_fn = collate_fn)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers = 15, collate_fn = collate_fn)
		return train_loader, test_loader
	
	else:
		dataset = PA_Dataset(args, type, normalization=args['normalization'], mode = mode)
		batch_size = args['batch_size']
		collate_fn = PA_Collator(batch_size, mode)
		plot_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers = 15, collate_fn = collate_fn)
		return plot_loader
