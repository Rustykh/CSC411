from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc.pilutil import imread,imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import shutil
import random


if __name__ == "__main__":

    act = ['vartan', 'baldwin', 'radcliffe', 'bracco', 'harmon', 'butler', 'carell', 'chenoweth', 'drescher', 'ferrera', 'gilpin', 'hader']
    female_act = ['drescher', 'ferrera', 'chenoweth', 'bracco', 'gilpin', 'harmon']
    male_act = ['baldwin', 'hader', 'carell', 'butler', 'radcliffe', 'vartan']
    if not os.path.isdir('training'):
        os.makedirs('training')
    if not os.path.isdir('test'):
        os.makedirs('test')
    if not os.path.isdir('validation'):
        os.makedirs('validation')

    for i in act:
        j = 0
        if i in female_act:
            gender = "female"
        if i in male_act:
            gender = "male"
        while (j <70):
            filename = i + str(j) + '.jpg'
            filename1 = i + str(j) + '.jpeg'
            filename2 = i + str(j) + '.JPG'
            filename3 = i + str(j) + '.png'
            files = [filename, filename1, filename2, filename3]
            for filename in files:
                if os.path.isfile("cropped/" + gender +"/" + filename):
                    srcpath = "cropped/" + gender + "/" + filename
                    dstpath = os.path.join("training/", filename)
            shutil.copyfile(srcpath, dstpath)
            j+=1

        while (j < 80):
            filename = i + str(j) + '.jpg'
            filename1 = i + str(j) + '.jpeg'
            filename2 = i + str(j) + '.JPG'
            filename3 = i + str(j) + '.png'
            files = [filename, filename1, filename2, filename3]
            for filename in files:
                if os.path.isfile("cropped/" + gender +"/"+filename):

                    srcpath = "cropped/" + gender + "/" + filename
                    dstpath = os.path.join("validation/", filename)
            shutil.copyfile(srcpath, dstpath)
            j+=1


        while (j < 90):
            filename = i + str(j) + '.jpg'
            filename1 = i + str(j) + '.jpeg'
            filename2 = i + str(j) + '.JPG'
            filename3 = i + str(j) + '.png'

            files = [filename, filename1, filename2, filename3]
            for filename in files:
                if os.path.isfile("cropped/" + gender +"/"+ filename):

                    srcpath = "cropped/" + gender + "/" + filename
                    dstpath = os.path.join("test/", filename)
            shutil.copyfile(srcpath, dstpath)
            if i == 'gilpin' and j == 87:
                break
            j+=1
