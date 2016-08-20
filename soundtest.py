import wx
from pyo import *

s = Server().boot()
s.start()
sf = SfPlayer("C:\\Program Files\\WorldViz\\Vizard5\\resources\\quack.wav", speed=1, loop=True).out()