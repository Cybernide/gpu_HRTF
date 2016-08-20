import viz
import vizact
import os
import viztask
import gpusndObj

viz.setMultiSample(4)
viz.fov(60)
viz.go()

viz.addChild('dojo.osgb')
#wheel = viz.addChild('wheelbarrow.ive')
viz.MainView.move([0,0,-7])
viz.clearcolor(viz.SKYBLUE)



#Add collision
viz.collision(viz.ON)#Add the duck model. 
duck = gpusndObj.addNewgpusndObj('duck.cfg', pos=[0,0.5,5], euler=[180,0,0])

#Attach a sound to the model and pause it
quack = gpusndObj.AudioFile("C:\\Program Files\\WorldViz\\Vizard5\\resources\\buzzer.wav", loop=True)
viztask.waitTime(10)
quack.play()
quack.stop()

#Play it when you strike the spacebar.
#vizact.onkeydown( ' ', quack.play() )
#vizact.onkeydown('a',viz.setDebugSound3D,viz.TOGGLE)
#vizact.onkeydown( 'd', quack.pause)

#viz.Scene1.getChildren()

#def playAHRTFsound():
#	viz._ipcSend

