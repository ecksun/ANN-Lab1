#!/usr/bin/python
import numpy
import pylab
import mlp
import sys

# execfile("test_separated.py")
execfile("test_merged.py")

# classA = numpy.random.randn(50,2)+1
# classB = numpy.random.randn(50,2)-1


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

def do(nhidden, beta, momentum, eta, color = 'k'):
    print "running with values %f %f %f %f" % (nhidden, beta, momentum, eta)

    tron = mlp.mlp(classC, targetC, nhidden, beta, momentum, 'logistic')

    tron.mlptrain(classC, targetC, eta, 1000)

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

    pylab.contour ( xrange , yrange , indicator , ( 0.5 , ), colors=color )

# Leker med eta
# do(2, 1, 0.9, 0.8,'r')
# do(2, 1, 0.9, 0.5,'g')
# do(2, 1, 0.9, 0.1,'b')


#leker med momentum
# do(2, 1, 0.9, 0.5,'r')
# do(2, 1, 0.5, 0.5,'g')
# do(2, 1, 0.1, 0.5,'b')

#leker med beta
# do(2, 1, 0.9, 0.5,'r')
# do(2, 0.5, 0.9, 0.5,'g')
# do(2, 0.1, 0.9, 0.5,'b')


# leker med nhidden
do(1, 1, 0.9, 0.5,'r')
do(2, 1, 0.9, 0.5,'g')
do(3, 1, 0.9, 0.5,'b')

pylab.show ( )
