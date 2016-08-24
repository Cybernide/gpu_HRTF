import viz
import vizmat
import vizconfig
import wave
import threading
from numpy import *
#from scipy import *
import pyaudio
import time
import math



def addNewgpusndObj(*args, **kwargs):
        """ make a 3D sound object"""
        newobj = gpusndObj(*args, **kwargs)
        return newobj
        
class gpusndObj(viz.VizNode):
    """ 3D sound object with an associated sound file"""
    def __init__(self, *args, **kwargs): 
        node = viz.addChild(*args, **kwargs)
        viz.VizNode.__init__(self, node.id)
        self.noise = None
        
    def setnoise(self, file, duration):
        self.noise = AudioFile(file, duration)
        
    
    def make3Dsnd(self, file, duration):
        '''1. Get location of self, and location of sound source.
        2. Determine elevation and azimuth of sound. 
        3. Magic convolutions.
        4. Stream sound.
        5. Profit.'''
        
        #obtain elevation and azimuth
        '''Currently because of Vizard's MainView positioning
        idiosyncracies, calculations using the y-axis are wonky
        but mostly close enough.'''
        me = viz.MainView.getPosition()
        src = self.getPosition()
        print me, src
        diffx, diffy, diffz = src[0]-me[0], src[1]-me[1], src[2]-me[2]
        elev = math.degrees(math.atan2(diffy, diffz))
        azi = math.degrees(math.atan2(diffx, diffz))
        print elev, azi
        readKEMAR(elev, azi)
        
        
class AudioFile(threading.Thread):
    chunk = 1024

    def __init__(self, filename, duration):
        """ Initialize audio stream""" 
        
        super(AudioFile, self).__init__()
        self.loop = True
        self.filename = filename
        self.duration = duration
        
    def run(self):
        """ Execute PyAudio """
        self.wf = wave.open(self.filename, 'rb')
        self.p = pyaudio.PyAudio()

        """ Loop through file """
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
            )
            
        # play sound    
        data = self.wf.readframes(self.chunk)
        while self.duration >= 0:
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)
            if data == '':
                self.wf.rewind()
                data = self.wf.readframes(self.chunk)
            time.sleep(1)
            self.duration -= 1
                
        self.stream.close()
        self.p.terminate()
                
    def play(self):
        self.start()
        
    def stop(self):
        self.loop = False
        
def readKEMAR(elev, azi):
    '''convert elevation and azimuth values into
    something usable by compressed KEMAR data
    notes: elevation goes from 90 to -40
    azimuth from 0 to 180 (left to right)'''
    fl_elev = int(round(elev, -1))
    if fl_elev > 90:
        fl_elev = 90
    if fl_elev < -40:
        fl_elev = -40
        
    '''Set increment of azimuth based on elevation
    This is horrible and ugly, but the KEMAR data files
    are named this way. It sucks. Royally.
    Shamelessly adapted from http://web.uvic.ca/~adambard
    Thank you so much, it saved me a lot of work.'''
        
    # Files in elevation increments of 10
    if abs(fl_elev) < 30:
        incr = 5
    elif abs(fl_elev) == 30:
        incr = 6
    elif abs(fl_elev) == 40:
        incr = 6.43
        opts = [0, 6, 13, 19, 26, 32, 29, 45, 51, 58, 64, 71, 77, 84, 90, 96, 103, 109, 116, 122, 129, 135, 141, 148, 154, 161, 167, 174, 180]
    elif fl_elev == 50:
        incr = 8
    elif fl_elev == 60:
        incr = 10
    elif fl_elev == 70:
        fl_incr = 15
    elif fl_elev == 80:
        fl_incr = 30
    elif fl_elev == 90:
        incr = 0
    flip = False
            
    #Constrain azimuth to 180 in front of view
    if azi > 180:
        azi = azi - 180
    if azi < -180:
        azi = azi + 180

    #If negative, flip left/right ears
    if azi < 0:
        azi = abs(azi)
        flip = True
        
    if abs(fl_elev) == 40:
        incr = 6.43
        num = incr
        while azi > num:
            num = num + incr
            
        azi = str(int(round(num)))
        #special case for non-integer increment
    elif azi != 0:
        while azi % incr > 0:
            azi = azi + 1
    
    if int(azi) < 100:
        azi = "0" + str(int(azi))
    if int(azi) < 10:
        azi = "00"+ str(int(azi))
        
    fl_KEMAR = "compact/elev"+str(fl_elev)+"/H"+str(fl_elev)+"e"+str(azi)+"a.wav"
    print fl_KEMAR
    ht = wave.open(fl_KEMAR, 'r')
    '''
    The process of splitting the left and right channels is
    almost entirely taken from:
    https://rsmith.home.xs4all.nl/miscellaneous/filtering-a-sound-recording.html
    Thank you, Roland.
    '''
    size = 44100 # Read and process 1 second at a time.
    data = fromstring(ht.readframes(size), dtype=int16)
    left, right = data[0::2], data[1::2]
    if flip:
        htr, htl= data[0::2], data[1::2]
    else:
        htl, htr = data[0::2], data[1::2]
        
    return htl, htr