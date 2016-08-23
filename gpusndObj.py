import viz
import vizmat
import vizconfig
import wave
import threading
from numpy import *
from scipy import *
import pyaudio
import time


#
#class gpusndObj(viz.VizNode):
#	def __init__(self, *args, **kwargs):
#		viz.addChild(*args, **kwargs)

def addNewgpusndObj(*args, **kwargs):
        newobj = gpusndObj(*args, **kwargs)
        return newobj
        
class gpusndObj(viz.VizNode): 
    def __init__(self, *args, **kwargs): 
        node = viz.addChild(*args, **kwargs)
        viz.VizNode.__init__(self, node.id)
        self.noise = None
        
    def setnoise(self, file, duration, pos):
        return AudioFile(file, duration, pos)
        
    
    def play3Dsnd(self, file, duration, pos):
        '''1. Get location of self, and location of sound source.
        2. Determine elevation and azimuth of sound. 
        3. Magic convolutions.
        4. Stream sound.
        5. Profit.'''
        me = self.getPosition()
        src = pos
        diffx = abs(me[0]-src[0])
        diffy = (me[1]-src[1])
        diffz = abs(me[2]-src[2])
        
        return
        
class AudioFile(threading.Thread):
    chunk = 1024

    def __init__(self, file, duration, pos):
        """ Initialize audio stream""" 
        
        super(AudioFile, self).__init__()
        self.loop = True
        self.file = file
        self.duration = duration
        self.azimuth = pos[0]
        self.elevation = pos[1]
        print viz.MainView.getPosition()
        
    def run(self):
        """ Execute PyAudio """
        self.wf = wave.open(self.file, 'rb')
        self.p = pyaudio.PyAudio()

        """ Loop through file """
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
            )
            
            
        data = self.wf.readframes(self.chunk)
        while self.duration >= 0:
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)
            if data == '':
                self.wf.rewind()
                data = self.wf.readframes(self.chunk)
            time.sleep(1)
            self.duration -= 1
        #self.loop = False
                
        self.stream.close()
        self.p.terminate()
                
    def play(self):
        self.start()
        
    def stop(self):
        self.loop = False
    