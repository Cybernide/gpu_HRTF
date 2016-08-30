These classes extend Vizard's VizNode class. Once instantiated,
a sound in the form of a wav file can be attached to the object
and given that a VizNode object has a location in the Viz 
environment, a head-related transfer function may be
applied to the sound.

Makes use of a multitude of Python modules, most notably NumPy
for the math (silly SciPy didn't want to build on my system),
pyAudio for playing the audio itself, and pyCUDA for making the
signal processing (a little) less arduous.

As a note, I chose to write a naive convolution kernel so that
I could practice my CUDA programming and learn a few things about
memory and hardware. In general, this project gave me more
experience with software engineering procedures and practices, as 
well as some practice with the fundamentals of signal processing.

Needless to say, I think I might have a better appreciation of
FFT. I can easily speed up the HRTF calculations with cuFFT.

This is a final project for a software engineering course. 
Things are not necessarily efficient, but they are educational!

To use, simply dump the contents of this folder anywhere you
wish and open any of the *sndTest*.py files. Currently only
accepts 44kHz, int16 uncompressed .wav files, but can be changed
pretty easily.

gpusndObj uses PyCUDA and GPU-accelerated naive convolution.
nsndObj uses NumPy GPU-only convolution.
gpusndObj2 uses GPU-accelerated shared memory convolution.
isndObj is iterative CPU-only convolution.

NOTE: gpusndObj2 is experiencing issues with channel separation.
isndObj has a lot of noise and distortion -- more than the other
modules. I've included them anyway to demonstrate differences in
processing speed between all these approaches.