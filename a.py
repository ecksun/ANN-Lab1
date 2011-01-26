#!/usr/bin/python
import numpy
import pylab
import mlp
import sys

test = "test_merged.py"
# test = "test_separated.py"

execfile(test)

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

def do(beta, eta, iterations, color = 'k'):
    nhidden = 2
    momentum = 0.9
    print "%s_beta-%f_eta-%f_iterations-%f" % (test, beta, eta, iterations)

    tron = mlp.mlp(classC, targetC, nhidden, beta, momentum, 'logistic')

    tron.mlptrain(classC, targetC, eta, iterations)

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

#0126 185826 <@tote> beta, learning rate, iterations, step_size(????)



# Leker med eta
# do(1, 0.1, 1000, 'r')
# do(1, 0.25, 1000, 'pink')
# do(1, 0.5, 1000, 'g')
# do(1, 0.9, 1000, 'b')

# leker med beta
# do(0.1, 0.25, 1000, 'r')
# do(0.5, 0.25, 1000, 'g')
# do(1, 0.25, 1000, 'b')

# leker med iterationer
do(1, 0.25, 10, 'r')
do(1, 0.25, 1000, 'g')
do(1, 0.25, 100000, 'b')

pylab.show ( )
