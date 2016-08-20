import wave
import pyaudio
if __name__ == '__main__':
		'''1. Get location of self, and location of sound source. 
		2. Determine elevation and azimuth of sound. 
		3. Magic convolutions.
		4. Stream sound.
		5. Profit.'''
		# open the file for reading.
		# length of data to read.
		chunk = 1024
		wf = wave.open('C:\\Program Files\\WorldViz\\Vizard5\\resources\\buzzer.wav', 'rb')

		# create an audio object
		p = pyaudio.PyAudio()

		# open stream based on the wave object which has been input.
		stream = p.open(
		format = p.get_format_from_width(wf.getsampwidth()),
		channels = wf.getnchannels(),
		rate = wf.getframerate(),
		output = True)
		# read data (based on the chunk size)
		data = wf.readframes(chunk)

		# play stream (looping from beginning of file to the end)
		while data != '':
			# writing to the stream is what *actually* plays the sound.
			stream.write(data)
			data = wf.readframes(chunk)

		# cleanup stuff.
		stream.close()    
		p.terminate()