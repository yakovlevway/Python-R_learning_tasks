                                            ###############################################################
                                            ###############################################################
                                            ############     Yakovlev Anton 451 group     #################
                                            ###############################################################
                                            
                                            ########       1 Zadanie (*)   ########            

from scipy.interpolate import lagrange
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import pi
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

c1, c2, m1, m2 = 1, 7, 0, 4
Summirovanie = c1 + c2 + m1 + m2
XLin = np.linspace(0, 1, 10)
Ylog = (Summirovanie) * np.log(np.pi - XLin + 1)**(np.log(XLin+1))
u = 10
poly = lagrange(XLin, Ylog)
XPLin = np.linspace(0,1,100)
YP = poly(XPLin)
XLinS = np.linspace(0, 1, 100) 
YSlog = (Summirovanie) * np.log(np.pi - XLinS + 1)**(np.log(XLinS+1))
fig, ax = plt.subplots()
ax.plot(XLin, Ylog, 'o') 
ax.plot(XPLin, YP, 'yellow', alpha= 0.5, linewidth = 4) 
ax.plot(XLinS, YSlog, 'purple') 
ax.grid(True)
plt.show()

                                            ########       1 Zadanie (**)   ######## 


from scipy.interpolate import lagrange
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import pi
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

b1, b2, b3, b4 = 0, 6, 1, 4
sub = b1 + b2 + b3 + b4
XLin = np.linspace(0, 1, 10)
Ylog = sub * abs(XLin - 0.5)
poly = lagrange(XLin, Ylog)
XPLin = np.linspace(0,1,100)
YP = poly(XPLin)
XLinS = np.linspace(0, 1, 100) 
YSlog = sub * abs(XLinS - 0.5)
fig, ax = plt.subplots()
ax.plot(XLin, Ylog, 'o') 
ax.plot(XPLin, YP, 'yellow', alpha= 0.5, linewidth = 4) 
ax.plot(XLinS, YSlog, 'purple') 
ax.grid(True)
plt.show()

                                            ########       1 Zadanie (***)   ########
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
xlins = np.linspace(0, 1, 10)
xlins[0] = 0.01
xtwo = np.linspace(0, 0.99, 100)
u = 901
T = 0
c1, c2, m1, m2 = 1, 7, 0, 4
b1, b2, b3, b4 = 0, 6, 1, 4
Gq = (c1 + c2 + m1 + m2) * np.log(np.pi - xlins + 1)**(np.log(xlins+1))
Gw = (b1 + b2 + b3 + b4) * abs(xlins - 0.5)
Ge = (c1 + c2 + m1 + m2) * np.log(np.pi - xtwo + 1)**(np.log(xtwo+1))

for k in range(u):
    T += (np.sin(np.pi*(u*xtwo-k))/(np.pi*(u*xtwo-k)))*Ge

plt.figure(figsize=(17, 17))
plt.plot(xtwo, Ge, color="red", alpha= 0.4, linewidth = 6, label = "f(xlins)")     
plt.plot(xtwo, T, color="gray", alpha= 1, linewidth = 3, label = "sinc")  
plt.legend()
plt.grid()
plt.show()


                                            ########       1 Zadanie (****)   ########

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
xlins = np.linspace(0, 1, 11)
xlins[0] = 0.01
xtwo = np.linspace(0, 1, 100)
u = 11
su = 0
c1, c2, m1, m2 = 1, 7, 0, 4
b1, b2, b3, b4 = 0, 6, 1, 4
Gq = (c1 + c2 + m1 + m2) * np.log(np.pi - xlins + 1)**(np.log(xlins+1))
Gw = (b1 + b2 + b3 + b4) * abs(xlins - 0.5)
Ge = (c1 + c2 + m1 + m2) * np.log(np.pi - xtwo + 1)**(np.log(xtwo+1))
Gr = (b1 + b2 + b3 + b4) * abs(xtwo - 0.5)

for k in range(u):
    su += (np.sin(np.pi*(u*xlins-k))/(np.pi*(u*xlins-k)))*Gw
    
plt.plot(xtwo, Gr, color="red", alpha= 0.4, linewidth = 6, label = "f(xlins)")     
plt.plot(xlins, su, color="gray", alpha= 1, linewidth = 3, label = "sinc") 
plt.legend()
plt.grid()
plt.show()



                                            ########       1 Zadanie (*****)   ########
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
xlins = np.linspace(0, 1, 11)
xtwo = np.linspace(0, 1, 100)
u = 10
su = 0
c1, c2, m1, m2 = 1, 7, 0, 4
b1, b2, b3, b4 = 0, 6, 1, 4
def y12(xlins):
    y111 = (c1 + c2 + m1 + m2) * np.log(np.pi - xlins + 1)**(np.log(xlins+1))
    return y111
Gq = (c1 + c2 + m1 + m2) * np.log(np.pi - xlins + 1)**(np.log(xlins+1))
Gw = (b1 + b2 + b3 + b4) * abs(xlins - 0.5)
Ge = (c1 + c2 + m1 + m2) * np.log(np.pi - xtwo + 1)**(np.log(xtwo+1))
for i in range(1, 11):
    x2 = np.linspace(xlins[i-1], xlins[i], 100)
    ss = y12(xlins[i-1]) * ((xlins[i] - x2)/(xlins[i]-xlins[i-1])) + (y12(xlins[i]) * (x2 - xlins[i-1])/(xlins[i]-xlins[i-1]))
    plt.plot(x2, ss, color = "gray")

plt.plot(xtwo, Ge,color="red", alpha= 0.4, linewidth = 7, label = "f(xlins)")
plt.plot(xlins,Gq,'o', markersize = 4)
plt.legend()
plt.grid()
plt.show


                                            ########       1 Zadanie (******)   ########

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
xlins = np.linspace(0, 1, 11)
xtwo = np.linspace(0, 1, 100)
u = 10
su = 0
b1, b2, b3, b4 = 0, 6, 1, 4
def y12(xlins):
    y111 = (b1 + b2 + b3 + b4) * abs(xlins - 0.5)
    return y111
Gw = (b1 + b2 + b3 + b4) * abs(xlins - 0.5)
Gr = (b1 + b2 + b3 + b4) * abs(xtwo - 0.5)
for i in range(1, 11):
    x2 = np.linspace(xlins[i-1], xlins[i], 100)
    ss = y12(xlins[i-1]) * ((xlins[i] - x2)/(xlins[i]-xlins[i-1])) + (y12(xlins[i]) * (x2 - xlins[i-1])/(xlins[i]-xlins[i-1]))
    plt.plot(x2, ss, color = "gray")

plt.plot(xtwo, Gr,color="red", alpha= 0.4, linewidth = 7, label = "f(xlins)")
plt.plot(xlins,Gw,'o', markersize = 4)
plt.grid()
plt.legend()
plt.show


                                            ########       2 Zadanie (Koef korel po form)   ######## 


from numpy import arange, cos, sin, pi
from math import sqrt, log, exp
import numpy as np

c1, c2, m1, m2 = 1, 7, 0, 4
b1, b2, b3, b4 = 0, 6, 1, 4
u = 10
XLin = []
for k in range(u):
    xk = 0.001 * cos((b1 + b2 + b3 + b4) * k) + exp(0.00001 * (c1 + c2 + m1 + m2) * k) * sin(k / (c1 + c2 + m1 + m2))
    XLin.append(xk)
print("Временной ряд XLin:", XLin)
XPLin = []
for k in range(u):
    xk = 0.001 * cos((b1 + b2 + b3 + b4) * (k - round(pi * (c1 + c2 + m1 + m2)))) + exp(1) ** (
                0.00001 * (c1 + c2 + m1 + m2) * (k - round(pi * (c1 + c2 + m1 + m2)))) * sin(
        (k - round(pi * (c1 + c2 + m1 + m2))) / (c1 + c2 + m1 + m2))
    XPLin.append(xk)
print("Временной ряд XPLin:", XPLin)
XLinS = []
for k in range(u):
    xk = 0.001 * cos((b1 + b2 + b3 + b4) * (k - round(2 * pi * (c1 + c2 + m1 + m2)))) + exp(1) ** (
                0.00001 * (c1 + c2 + m1 + m2) * (k - round(2 * pi * (c1 + c2 + m1 + m2)))) * sin(
        (k - round(2 * pi * (c1 + c2 + m1 + m2))) / (c1 + c2 + m1 + m2))
    XLinS.append(xk)
print("Временной ряд XLinS:", XLinS)
ccPirs = np.corrcoef(XLin, XPLin)
ccPirstw = np.corrcoef(XLin, XLinS)
print("Коэффицент корреляции между XLin и XPLin", ccPirs[0, 1])
print("Коэффицент корреляции между XLin и XLinS", ccPirstw[0, 1])


