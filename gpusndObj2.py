"""
This class extends Vizard's VizNode class. Once instantiated,
a sound in the form of a wav file can be attached to the object
and given that a VizNode object has a location in the Viz 
environment, a head-related transfer function may be
applied to the sound.

Notable imports are: pyAudio for playing the audio itself 
and pyCUDA for making the signal processing (a little) less 
arduous.
"""

# Yes, there are many dependencies.
# I am not a very independent module.
import viz
import wave
import threading
import numpy
import pyaudio
import time
import math
from pycuda import gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def addNewgpusndObj(*args, **kwargs):
        """ make a 3D sound object based off the VizNode class"""
        newobj = gpusndObj(*args, **kwargs)
        return newobj

class gpusndObj(viz.VizNode):
    """ 3D sound object with an associated sound file"""
    def __init__(self, *args, **kwargs): 
        node = viz.addChild(*args, **kwargs)
        viz.VizNode.__init__(self, node.id)
        self.noise = None
        
    def setNoise(self, filename, duration):
        """Attaches a wave file to the object"""
        print(filename)
        vect = list(self.getAngles())
        dat= list(process3D(vect[0], vect[1], filename))
        fl = write2stereo(dat[0],dat[1],dat[2], str(vect[0]) + str(vect[1]))
        self.noise = AudioFile('snd3d.wav', duration)
        
    # Currently because of Vizard's MainView positioning
    # idiosyncracies, calculations using the y-axis are wonky
    # but mostly close enough. I add 1.0 to the y-coordinate
    # to compensate somewhat.
    def getAngles(self):
        ''' Returns elevation and azimuth of the object relative
        to Vizard's MainView.'''
        
        me = viz.MainView.getPosition()
        src = self.getPosition()
        diffx, diffy, diffz = (src[0]-me[0],
            (src[1]+1.0)-me[1], src[2]-me[2])
        elev = math.degrees(math.atan2(diffy, diffz))
        azi = math.degrees(math.atan2(diffx, diffz))
        
        print ('Elevation at: ' + str(elev) + 
            ' \n' + 'Azimuth: ' + str(azi))
        
        return elev, azi
        
        
class AudioFile(threading.Thread):
    """The sound capabilities attached to a gpusndObj"""
    chunk = 1024

    def __init__(self, filename, duration):
        """ Initialize audio stream""" 
        super(AudioFile, self).__init__()
        self.loop = True
        self.filename = filename
        self.duration = duration
        
    def run(self):
        """ Execute and play stream"""
        self.wf = wave.open(self.filename, 'rb')
        self.p = pyaudio.PyAudio()

        # Open pyAudio stream
        self.stream = self.p.open(
            format = self.p.get_format_from_width(
            self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
            )
            
        # play and loop sound   
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
        """Start audio stream from a sndObj"""
        self.start()
        
    def stop(self):
        """Stop audio stream from a sndObj"""
        self.loop = False

        
# Notes on KEMAR data: elevation goes from 90 to -40,
# azimuth from 0 to 180 (left to right). The increments
# are irregular, though, so reading the files in
# results in some really ugly code.
def readKEMAR(elev, azi):
    """Convert elevation and azimuth values into
    string to read compressed KEMAR data file names.
    Returns impulse responses as two HRTF filter 
    arrays for each ear."""
    
    
    
    # Elevation rounded to the nearest 10
    fl_elev = int(round(elev, -1))
    # Values between 90 and -40
    if fl_elev > 90:
        fl_elev = 90
    if fl_elev < -40:
        fl_elev = -40
        
    # Set increment of azimuth based on elevation
    # This is horrible and ugly, but the KEMAR data files
    # are named this way. It sucks. Royally.
    # Shamelessly adapted from http://web.uvic.ca/~adambard
    # Thank you so much, it saved me a lot of work.
        
    # Files in elevation increments of 10.
    if abs(fl_elev) < 30:
        incr = 5
    elif abs(fl_elev) == 30:
        incr = 6
    elif abs(fl_elev) == 40:
        incr = 6.43
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
            
    # Constrain azimuth to 180 degrees in front of view.
    if azi > 180:
        azi = azi - 180
    if azi < -180:
        azi = azi + 180

    # If negative, flip left/right ears.
    if azi < 0:
        azi = abs(azi)
        flip = True
    
    # Yes, this is ugly, but I couldn't
    # think of a better way to do this.
    if abs(fl_elev) == 40:
        incr = 6.43
        num = incr
        while azi > num:
            num = num + incr
        # so elevation is 40, we have our azimuth!
        azi = str(int(round(num)))
        
    #special case for non-integer increment
    elif azi != 0:
        azi = int(incr * round(float(azi)/incr))
        
    # Finally, turn this mess into something
    # that can access the data.
    if int(azi) < 100:
        azi = '0' + str(int(azi))
    if int(azi) < 10:
        azi = '00'+ str(int(azi))
        
    fl_KEMAR = (
        'compact/elev'+str(fl_elev)+'/H'+str(fl_elev)+'e'+str(azi)+
        'a.wav'
        )
    ht = wave.open(fl_KEMAR, 'r')
    
    # Compact KEMAR is in stereo, so we have to extract
    # left and right channels as mono.
    # The process of splitting the left and right channels is
    # almost entirely taken from:
    # https://rsmith.home.xs4all.nl/miscellaneous/filtering-a-
    # sound-recording.html
    # Thank you, Roland.
    
    data = numpy.fromstring(
        ht.readframes(ht.getframerate()), 
        dtype=numpy.int16
        )

    # Extract left and right channels from the file.
    if flip:
        htr, htl= data[0::2], data[1::2]
    else:
        htl, htr = data[0::2], data[1::2]
    return htl, htr
    
def process3D(elev, azi, filename):
    """ Given elevation and azimuth, take the file given by 'filename'
    and process it so that it returns spatialized audio. Uses PyCUDA
    to convolve signal and filters. """
    
    # read-in data
    src = wave.open(filename, 'r')
    params = list(src.getparams()) # saved for file reconstruction
    htl, htr = readKEMAR(elev, azi)
    src_d = numpy.fromstring(
        src.readframes(src.getframerate()), dtype=numpy.int16)
    src_d = src_d/(max(src_d))
        
    # Begin GPU-accelerated convolution
    
    # If the filter array has an even
    # number of elements, add 0 to the
    # beginning to produce an odd-
    # element convolution window.
    if htl.size % 2 == 0:
        htl = numpy.append(0, htl)
    if htr.size % 2 == 0:
        htr = numpy.append(0, htr)
    
    # Reverse filter array for convolving
    # with signal.
    htl = htl[::-1]
    htr = htr[::-1]

    # Make htl and htr C-contiguous so that
    # they can be read into PyCUDA.
    htl = numpy.ascontiguousarray(htl)
    htr = numpy.ascontiguousarray(htr)

    # Begin CUDA module
    mod = SourceModule("""
        __global__ void convKernel(int *htl, int *htr, int *sig, int *outl, int *outr, int sigsize) {
        const int TILE_SIZE = 1024;
        const int MASK_W = 129;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ int  sig_l[TILE_SIZE];

  sig_l[threadIdx.x] = sig[i];

  __syncthreads();

  int This_start_l = blockIdx.x * blockDim.x;
  int Next_start_l = (blockIdx.x + 1) * blockDim.x;
  int sig_start_point = i - (MASK_W/2);
  int outl_val= 0;
  for (int j = 0; j < MASK_W; j ++) {
     int sig_index_l = sig_start_point + j;
     if (sig_index_l >= 0 && sig_index_l < sigsize) {
       if ((sig_index_l >= This_start_l)
         && (sig_index_l < Next_start_l))
         outl_val += sig_l[threadIdx.x+j-(MASK_W/2)]*htl[j];
       } else {
         outl_val += sig_l[sig_index_l] * htl[j];
       }
     }
  outl[i] = outl_val;
  
int k = blockIdx.y*blockDim.y + threadIdx.y;
  __shared__ int  sig_r[TILE_SIZE];

  sig_r[threadIdx.y] = sig[k];

  __syncthreads();

  int This_start_r = blockIdx.y * blockDim.y;
  int Next_start_r = (blockIdx.y + 1) * blockDim.y;
  int sig_start_r = k - (MASK_W/2);
  int outr_val = 0;
  for (int m = 0; m < MASK_W; m ++) {
     int sig_index_r = sig_start_r + m;
     if (sig_index_r >= 0 && sig_index_r < sigsize) {
       if ((sig_index_r >= This_start_r)
         && (sig_index_r < Next_start_r))
         outr_val += sig_r[threadIdx.y+m-(MASK_W/2)]*htr[k];
       } else {
         outr_val += sig[sig_index_r] * htr[k];
       }
     }
  outr[k] = outr_val;  
  
  }
    """)
    
    
    
    # Initialize arrays for outputting results of CUDA kernel
    dev_l_out= numpy.zeros(src_d.size, dtype=numpy.int16, order='C')
    dev_r_out= numpy.zeros(src_d.size, dtype=numpy.int16, order='C')
    
    # Call kernel
    
    func = mod.get_function("convKernel")
    from timeit import default_timer
    startTime = default_timer()
    func(cuda.In(htl), 
        cuda.In(htr), 
        cuda.In(src_d),
        cuda.InOut(dev_l_out), 
        cuda.InOut(dev_r_out), 
        #numpy.int32(htl.size),
        #numpy.int32(htr.size),
        numpy.int32(src_d.size), 
        grid = (2,2), block=(32,32,1))
    endTime = (default_timer() - startTime)
    print ('Elapsed running time was ' + str(endTime))
    
    '''func(
        #cuda.In(htl), 
        cuda.In(htr), 
        cuda.In(src_d),
        #cuda.InOut(dev_l_out), 
        cuda.InOut(dev_r_out), 
        #numpy.int32(htl.size),
        #numpy.int32(htr.size),
        #numpy.int32(src_d.size), 
        grid = (2,2), block=(32,32,1))'''
        
    print dev_l_out == dev_r_out
    return dev_l_out, dev_r_out, params

def write2stereo(left, right, params, pos_string):
    """ Convert two numpy arrays into a stereo audio file. """
    ofl = wave.open('snd3d' + pos_string + '.wav','w')
    params[0] = 2 # specify 2 audio channels
    ofl.setparams(tuple(params))
    # Stereo .wav files are interleaved, thus we interleave
    # the values of our numpy array!
    ostr = numpy.column_stack((left,right)).ravel()
    ofl.writeframes(ostr.tostring())
    ofl.close()
    return 'snd3d' + pos_string + '.wav'