# todo; use spotipy after train

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import time
from multiprocessing import Process

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

'''
for file in glob('audio/*.mp3'):
    f = ('audio/'+file[6:])
    song = AudioSegment.from_mp3(f)
    play(song)
'''
song = AudioSegment.from_file('test.mp3', format="mp3")