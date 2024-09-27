##################################################################
## getModel.py                                                  ##
## This file is used to get the prediction model based on the   ##
## input arguments.                                             ##
## The model can be selected by the option argument.            ##
## 1. RidgeModel : Linear Regression based prediction model     ##
## 2. IHE_CLSTM : CNN-LSTM based prediction model               ## 
## 3. CircuitMixer : Mixer based prediction model               ##
##################################################################
## Author: Jaeseung Lee                                         ##
## Date: September 2024                                         ##
## Affiliation: POSTECH CSDL, South Korea                       ##
##################################################################

from model import *


## Option 1: RidgeModel, Option 2: IHE_CLSTM (2nd years), Option 3: CircuitMixer (3rd years)
def get_model(args, num_param, option=1):        
    if option == 1:
        model = RidgeModel(
            hidden_size = int(args['hidden_size']),
            num_param = num_param,
            num_output = 1,
        )

    if option == 2:
        model = IHE_CLSTM(
            hidden_size = int(args['hidden_size']),
            window_size = int(args['window_size']),
            lstm_num_layers = int(args['num_layers']),
            batch_size = int(args['batch_size']),
            num_param = num_param,
            num_output = 1,
        )
    
    if option == 3:
        model = CircuitMixer(
            data_length = args['window_size'],
            pred_length = int((args['end'] - args['start']) * args['resolution']) - args['window_size'],
            num_blocks = args['num_layers'],
            input_channel = 1,
            hidden_channel = args['hidden_channel'],
            hidden_size= args['hidden_size'],
            param_length = num_param - 1,
            output_channel = 1,
        )

    return model.cuda()
