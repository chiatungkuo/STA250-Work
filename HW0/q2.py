#! /usr/bin/env python

from pylab import *

x = 2 * pi * rand(1000)
y = rand(1000)

# polar transformation
u = y * cos(x)
v = y * sin(x)
r = sqrt(u**2 + v**2)
figure()
plot(u, v, 'o')
figure()
hist(r)

raw_input("Press Enter to exit...")
