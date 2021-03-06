﻿import viz
import vizact
import os
import viztask
import nsndObj

viz.setMultiSample(4)
viz.fov(60)




world = viz.addChild('dojo.osgb')
pic = viz.addTexture('lake3.jpg', flags=viz.LOAD_ASYNC)

 
viz.go()

# change any of these positions to test out the HRTF.
viz.MainView.move([0,0,-7])
duck1 = nsndObj.addNewsndObj('duck.cfg', pos=[2,1,-6], euler=[180,0,0])

def onModelLoad(e):
	if e.status == viz.ASYNC_SUCCESS:
		dummy = viz.addChild('dojo.osgb', flags=viz.LOAD_ASYNC)
		vizact.onAsyncLoad(dummy, onLoad)

		
vizact.onAsyncLoad(pic, onModelLoad)		
		
def onLoad(e):
	if e.status == viz.ASYNC_SUCCESS:
		from timeit import default_timer
		startTime = default_timer()
		quack1 = duck1.setNoise("buzzer.wav", duration=2)
		duck1.noise.play()
		endTime = (default_timer() - startTime)
		#print ('Elapsed running time was ' + str(endTime) + ' ms.')
		viz.waitTime(4)
		duck1.remove()
		viz.quit()
