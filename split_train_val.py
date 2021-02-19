import os
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("--directorio_origen", type=str, default="data/custom/images", help="Directorio donde se encuentran todas las imagenes y las etiquetas")
parser.add_argument("--directorio_destino", type=str, default="data/custom", help="directorio donde se escribira train y test txt")
opt = parser.parse_args()

path = opt.directorio_origen
dr = "data/custom/labels"
ff = os.listdir(dr)
files = os.listdir(path)
random.shuffle(files)
train = files[:int(len(files)*0.9)]
val = files[int(len(files)*0.9):]

for item in ff:
    name = item.split('.')
    print(name[0])

# with open('{}/train.txt'.format(opt.directorio_destino), 'w') as f:
#     for item in train:
#         nameimg = item.split('.')
#         for item2 in ff:
#             name = item2.split('.')
#             if(name[0] == nameimg[0]):
#                 f.write("{}/{} \n".format(path, item))
#                 break

with open('{}/valid.txt'.format(opt.directorio_destino), 'w') as f:
    for item in val:
        nameimg = item.split('.')
        for item2 in ff:
            name = item2.split('.')
            # print(name[0],nameimg[0])
            if(name[0] == nameimg[0]):
                f.write("{}/{} \n".format(path, item))
                break