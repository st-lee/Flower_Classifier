import argparse

parser = argparse.ArgumentParser()
parser.add_argument('img_path', action = 'store')
parser.add_argument('checkpoint', action = 'store')

parser.add_argument('--top_k', action = 'store',
                    dest = 'top_k',
                    type = int, 
                    default = 5,
                    help = 'Set number of top classes which will be return')

parser.add_argument('--category_names', action = 'store',
                    dest = 'category_names',
                    type = str, 
                    default = 'cat_to_name.json',
                    help = 'Set a mapping of categories to real flowers name')

parser.add_argument('--gpu', action = 'store_true',
                    default = 'False',
                    dest = 'gpu',
                    help = 'Use GPU')

results = parser.parse_args()

def get_args():
    return results
