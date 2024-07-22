from model import *


def get_model(args, num_param, PE = 0, mode = 0, option=0):
    if option == 0:
        model = Model(args, num_param, PE, mode)
        
    if option == 1:
        model = FastCNNLSTM(args, num_param, 1, PE, mode)

    if option == 2:
        model = FastModel(args, num_param, 1, PE, mode)
    
    if option == 3:
        model = CircuitMixer(
            data_length = args['window_size'],
            pred_length = int((args['end'] - args['start']) * args['resolution']) - args['window_size'],
            num_blocks = args['num_layers'],
            input_channel = 1,
            hidden_channel = args['hidden_channel'],
            hidden_size= args['hidden_size'],
            param_length = num_param -1,
            output_channel = 1,
        ).cuda()

    return model
