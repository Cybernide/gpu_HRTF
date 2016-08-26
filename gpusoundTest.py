import viz
import vizact
import os
import viztask
import gpusndObj

viz.setMultiSample(4)
viz.fov(60)
viz.go()

viz.addChild('dojo.osgb')

#sets view position
viz.MainView.move([0,0,-7])

#new sound obj
duck = gpusndObj.addNewgpusndObj('duck.cfg', pos=[0,0,7], euler=[180,0,0])
quack = duck.setNoise("buzzer.wav", duration=4)
duck.noise.play()
#quack.pause()


