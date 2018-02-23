from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imresize, imread, imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import shutil
import sys



def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()



if __name__ == "__main__":
    if (len(sys.argv[1:]) != 2):
        print "Usage: python get_data.py male_filename female_filename"
        sys.exit(1)

    male_act = list(set([a.split("\t")[0] for a in open(sys.argv[1]).readlines()]))
    female_act = list(set([a.split("\t")[0] for a in open(sys.argv[2]).readlines()]))
    act = male_act + female_act
    if not os.path.isdir('uncropped'):
        os.makedirs('uncropped')

    if not os.path.isdir('cropped'):
        os.makedirs('cropped')

    if not os.path.isdir('cropped/male'):
        os.makedirs('cropped/male')

    if not os.path.isdir('cropped/female'):
        os.makedirs('cropped/female')
    for a in act:
        # print act
        if a in male_act:
            f = sys.argv[1]
            gender = "male"
        if a in female_act:
            f = sys.argv[2]
            gender = "female"
        name = a.split()[1].lower()
        i = 0
        for line in open(f):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                print line
                face_dim = line.split("\t")[4].split(',')
                face_dim = [int(j) for j in face_dim]
                # print(face_dim)
                #A version without timeout (uncomment in case you need to
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                try:
                    imarray = imread("uncropped/" + filename)
                except:
                    continue
                print "this is im array"
                print face_dim
                cropped = imarray[face_dim[1]:face_dim[3], face_dim[0]:face_dim[2]]
                resized = imresize(cropped, (32,32))
                if len(resized.shape) == 3:
                    grayscale = rgb2gray(resized)
                imsave("cropped/"+ gender + "/" + filename, grayscale)




                print filename
                i += 1
