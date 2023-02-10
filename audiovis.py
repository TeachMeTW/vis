# todo; use spotipy after train
# need ffmpeg
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

import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import TclError

import numpy as np
from opensimplex import OpenSimplex
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import struct
import pyaudio
import sys

CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100 


def play_song():
    for file in glob('audio/*.mp3'):
        f = ('audio/'+file[6:])
        song = AudioSegment.from_mp3(f)
        play(song)
    

def visual():
    fig, ax = plt.subplots(1, figsize=(15, 7))

    # pyaudio class instance
    p = pyaudio.PyAudio()

    # stream object to get data from microphone
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )

    # variable for plotting
    x = np.arange(0, 2 * CHUNK, 2)

    # create a line object with random data
    line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)

    # basic formatting for the axes
    ax.set_title('AUDIO WAVEFORM')
    ax.set_xlabel('samples')
    ax.set_ylabel('volume')
    ax.set_ylim(0, 255)
    ax.set_xlim(0, 2 * CHUNK)
    plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

    # show the plot
    plt.show(block=False)

    print('stream started')

    # for measuring frame rate
    frame_count = 0
    start_time = time.time()

    while True:
        
        # binary data
        data = stream.read(CHUNK)  
        
        # convert data to integers, make np array, then offset it by 127
        data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
        
        # create np array and offset by 128
        data_np = np.array(data_int, dtype='b')[::2] + 128
        
        line.set_ydata(data_np)
        
        # update figure canvas
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
            frame_count += 1
            
        except TclError:
            
            # calculate average frame rate
            frame_rate = frame_count / (time.time() - start_time)
            
            print('stream stopped')
            print('average frame rate = {:.0f} FPS'.format(frame_rate))
            break
            
        

class Terrain(object):
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """

        # setup the view window
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Terrain')
        self.window.setGeometry(0, 110, 1920, 1080)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()

        # constants and arrays
        self.nsteps = 1.3
        self.offset = 0
        self.ypoints = np.arange(-20, 20 + self.nsteps, self.nsteps)
        self.xpoints = np.arange(-20, 20 + self.nsteps, self.nsteps)
        self.nfaces = len(self.ypoints)

        self.RATE = 44100
        self.CHUNK = len(self.xpoints) * len(self.ypoints)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )

        # perlin noise object
        self.noise = OpenSimplex(13)

        # create the veritices array
        verts, faces, colors = self.mesh()

        self.mesh1 = gl.GLMeshItem(
            faces=faces,
            vertexes=verts,
            faceColors=colors,
            drawEdges=True,
            smooth=False,
        )
        self.mesh1.setGLOptions('additive')
        self.window.addItem(self.mesh1)

    def mesh(self, offset=0, height=2.5, wf_data=None):

        if wf_data is not None:
            wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', wf_data)
            wf_data = np.array(wf_data, dtype='b')[::2] + 128
            wf_data = np.array(wf_data, dtype='int32') - 128
            wf_data = wf_data * 0.04
            wf_data = wf_data.reshape((len(self.xpoints), len(self.ypoints)))
        else:
            wf_data = np.array([1] * 1024)
            wf_data = wf_data.reshape((len(self.xpoints), len(self.ypoints)))

        faces = []
        colors = []
        verts = np.array([
            [
                x, y, wf_data[xid][yid] * self.noise.noise2(x=xid / 5 + offset, y=yid / 5 + offset)
            ] for xid, x in enumerate(self.xpoints) for yid, y in enumerate(self.ypoints)
        ], dtype=np.float32)

        for yid in range(self.nfaces - 1):
            yoff = yid * self.nfaces
            for xid in range(self.nfaces - 1):
                faces.append([
                    xid + yoff,
                    xid + yoff + self.nfaces,
                    xid + yoff + self.nfaces + 1,
                ])
                faces.append([
                    xid + yoff,
                    xid + yoff + 1,
                    xid + yoff + self.nfaces + 1,
                ])
                colors.append([
                    xid / self.nfaces, 1 - xid / self.nfaces, yid / self.nfaces, 0.7
                ])
                colors.append([
                    xid / self.nfaces, 1 - xid / self.nfaces, yid / self.nfaces, 0.8
                ])

        faces = np.array(faces, dtype=np.uint32)
        colors = np.array(colors, dtype=np.float32)

        return verts, faces, colors

    def update(self):
        """
        update the mesh and shift the noise each time
        """

        wf_data = self.stream.read(self.CHUNK, exception_on_overflow=False)

        verts, faces, colors = self.mesh(offset=self.offset, wf_data=wf_data)
        self.mesh1.setMeshData(vertexes=verts, faces=faces, faceColors=colors)
        self.offset -= 0.05

    def start(self):
        """
        get the graphics window open and setup
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec()

    def animation(self, frametime=10):
        """
        calls the update method to run in a loop
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()
                
if __name__ == "__main__":        
    p1 = Process(target=play_song, args=())
    p1.start()
    t = Terrain()
    p2 = Process(target=t.animation(), args=())
    p2.start()
    

    #p2.join()
    #p1.join()