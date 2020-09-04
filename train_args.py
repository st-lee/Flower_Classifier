import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action = 'store')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_dir',
                    type = str,
                    default = './checkpoint.pth',
                    help = 'Set checkpoint dir')

parser.add_argument('--arch', action = 'store',
                    dest = 'arch',
                    type = str,
                    default = 'vgg16',
                    help = 'Set architecture')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'learning_rate',
                    type = float,
                    default = 0.001,
                    help = 'Set learning rate')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hidden_units',
                    type = int, 
                    default = 1024,
                    help = 'Set hidden layer number')

parser.add_argument('--epochs', action = 'store',
                    dest = 'epochs',
                    type = int, 
                    default = 5,
                    help = 'Set epochs')

parser.add_argument('--dropout', action = 'store',
                    dest = 'dropout',
                    type = int, 
                    default = 0.5,
                    help = 'Set epochs')

parser.add_argument('--gpu', action = 'store_true',
                    default = 'False',
                    dest = 'gpu',
                    help = 'Use GPU')

results = parser.parse_args()

def get_args():
    return results
