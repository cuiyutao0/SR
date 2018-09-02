import argparse
import time
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import  Scale

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name',default='2.jpg')
parser.add_argument('--model_name', default='netG_epoch_4_30.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

dir = '/home/lucas/Lab/SRGAN/SRGAN/data/test2'
files = os.listdir(dir)
for file in files:
    image = Image.open(dir+'/'+file)
    print(image.size)
    print(image.size[0])
    min_edge = image.size[0] if image.size[0] < image.size[1] else image.size[1]
    image_lr = Scale(min_edge//2, interpolation=Image.BICUBIC)(image)

    image = Variable(ToTensor()(image_lr), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.clock()
    out = model(image)
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    #out_img.save(dir + '/' + file[:-4] + '_generate' + str(UPSCALE_FACTOR) + '_init.jpg')
    #out_img = Scale(256, interpolation=Image.BICUBIC)(out_img)
    out_img.save(dir+'/' + file[:-4] + '_generate' + '.jpg')
    image_lr = Scale(min_edge*2 , interpolation=Image.BICUBIC)(image_lr)
    image_lr.save(dir + '/' + file[:-4] + '_init' + '.jpg')
