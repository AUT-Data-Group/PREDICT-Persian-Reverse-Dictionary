import pickle, json, numpy as np


def read_pickle(file_dir):
    with open(file_dir,'rb') as file:
        var = pickle.load(file)
    return var


def read_json(file_dir):
    with open(file_dir, 'r',encoding='utf-8') as file:
        var = json.load(file)
    return var

def read_lines(file_dir):
    with open(file_dir,'r',encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines

def read_csv_as_dict(file_dir):
    lines = read_lines(file_dir)
    lines = [line.split(',') for line in lines]
    dictionary = {key:value for (key, value) in lines}
    return dictionary

def write_line_to_file(line, file_dir):
    with open(file_dir, 'a+', encoding='utf-8') as file:
        file.write(line+'\n')

def init_log(log_obj, log_dir):
    log_begin = ''
    for (key, value) in log_obj.items():
        log_begin += key+' : '+value+'\n'
    with open(log_dir, 'w', encoding='utf-8') as file:
        file.write(log_begin)
        
def get_more_ram():
    a = np.zeros((5000000,1000000))
    return a

