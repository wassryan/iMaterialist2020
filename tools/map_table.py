import os
import pandas as pd
import json
import pickle

### create mapping from old class to new class ###
root = '/data/imaterialist2020'
map_file = root + '/attr_map'

with open(root+'/label_descriptions.json', 'r') as file:
    label_desc = json.load(file)

categories_df = pd.DataFrame(label_desc['categories'])
attributes_df = pd.DataFrame(label_desc['attributes'])

attr_values = attributes_df['id'].values
print(attr_values)
print(len(attr_values))

attr_dict = {
    "attr2new": {-1:0},
    "new2attr": {0:-1}} # newlabel: old label

for idx, value in enumerate(attr_values):
    # attr_dict[idx+1] = value
    attr_dict["attr2new"][value] = idx + 1
    attr_dict["new2attr"][idx+1] = value

def save_pkl(obj, pkl_name):
    with open(pkl_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(pkl_name):
    with open(pkl_name + '.pkl', 'rb') as f:
        return pickle.load(f)

save_pkl(attr_dict, map_file)
# xx = load_pkl(map_file)
print("Map Finish...")