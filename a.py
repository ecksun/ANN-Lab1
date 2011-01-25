import numpy
import pylab
import mlp

classA = numpy.random.randn(50,2)+1
classB = numpy.random.randn(50,2)*2-1

classC = numpy.concatenate((classA, classB))

# print classA

targetA = numpy.zeros((50,1))
targetB = numpy.ones((50,1))

targetC = numpy.concatenate((targetA, targetB))
# print targetC

pylab.plot(classA[:,0], classA[:,1], 'g.')
pylab.plot(classB[:,0], classB[:,1], 'r.')
# pylab.plot(classC[:,0], classC[:,1], 'r.')
# pylab.show()


tron = mlp.mlp(classC, targetC, 2, 1, 0.5)

tron.mlptrain(classC, targetC, 0.4, 1000)

xrange = numpy.arange( -4 , 4 , 0.1 )

yrange = numpy.arange(-4 , 4 , 0.1 )

xgrid,ygrid = numpy.meshgrid ( xrange , yrange )

noOfPoints = xgrid.shape[ 0 ] * xgrid.shape [ 1 ]

xcoords = xgrid.reshape( ( noOfPoints , 1 ) )
ycoords = ygrid.reshape( ( noOfPoints , 1 ) )
samples = numpy.concatenate( ( xcoords , ycoords) , axis=1)

ones = -numpy.ones ( xcoords.shape )

samples = numpy.concatenate ( ( samples , ones ) , axis =1)

indicator = tron.mlpfwd ( samples )
indicator = indicator.reshape ( xgrid . shape )

pylab.contour ( xrange , yrange , indicator , ( 0.5 , ) )
pylab.show ( )

