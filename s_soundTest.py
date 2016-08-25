﻿import viz
import vizact
import os
import viztask
import sndObj

viz.setMultiSample(4)
viz.fov(60)
viz.go()

viz.addChild('dojo.osgb')

#sets view position
viz.MainView.move([0,0,-7])


#new sound obj
duck = sndObj.addNewsndObj('duck.cfg', pos=[0,0,7], euler=[180,0,0])
duck.getAngles()
quack = duck.setNoise("buzzer.wav", duration=4)
duck.noise.play()