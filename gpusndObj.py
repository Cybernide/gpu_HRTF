import viz
import vizmat
import vizconfig
import wave
import threading
import pyaudio
import time

def addNewgpusndObj(*args, **kwargs):
	newobj = gpusndObj(*args, **kwargs)
	return newobj

class gpusndObj(viz.VizObject):
	def __init__(self, *args, **kwargs):
		viz.addChild(*args, **kwargs)
		
	def play3Dsnd(self, fl, mode):
		'''1. Get location of self, and location of sound source. 
		2. Determine elevation and azimuth of sound. 
		3. Magic convolutions.
		4. Stream sound.
		5. Profit.'''
		# open the file for reading.
		# length of data to read.
		
		return
		
class AudioFile(threading.Thread):
    chunk = 1024

    def __init__(self, file, duration):
        """ Initialize audio stream""" 
        
        super(AudioFile, self).__init__()
        self.loop = True
        self.file = file
        self.duration = duration
        
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



# Usage example for pyaudio
#a = AudioFile("C:\\Program Files\\WorldViz\\Vizard5\\resources\\buzzer.wav")
#a.play()
#a.close()