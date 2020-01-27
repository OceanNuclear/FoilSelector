from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
import unittest
from collapx import area_between_2_pts, Integrate
import numpy.random as rn
from scipy.integrate import quadrature # get gaussian quadrature
import openmc 

class AreaTest(unittest.TestCase):

    # doesn't need an init method

    def fullrange(self, scheme=1):
        PLOT = False
        # rn.seed(0)
        xydata = rn.randn(2,2)
        logx, logy=False, False
        if scheme>=2:
            logy, logx = [bool(int(i)) for i in bin(scheme-2)[2:].zfill(2)]
        if logx:
            xydata[:,0] = abs(xydata[:,0])
        if logy:
            xydata[:,1] = abs(xydata[:,1])
        xydata.sort(axis=0)
        xy1, xy2 = xydata
        func = openmc.data.Tabulated1D(*xydata.T, breakpoints=[2], interpolation=[scheme])
        if PLOT:
            x = np.linspace(xy1[0],xy2[0], 1000)
            plt.plot(x, func(x))
            plt.xticks([xy1[0], xy2[0]])
            plt.show()
        return np.isclose(quadrature(func, xy1[0], xy2[0], tol=1E-12)[0], area_between_2_pts(xy1, xy2, xy2[0], scheme))

    def test_scheme_1(self):
        assert self.fullrange(1)
        assert self.inbetween(1)
    def test_scheme_2(self):
        assert self.fullrange(2)
        assert self.inbetween(2)
    def test_scheme_3(self):
        assert self.fullrange(3)
        assert self.inbetween(3)
    def test_scheme_4(self):
        assert self.fullrange(4)
        assert self.inbetween(4)
    def test_scheme_5(self):
        assert self.fullrange(5)
        assert self.inbetween(5)
    
    def inbetween(self, scheme=1):
        PLOT = False
        # rn.seed(0)
        xydata = rn.randn(2,2)
        logx, logy=False, False
        if scheme>=2:
            logy, logx = [bool(int(i)) for i in bin(scheme-2)[2:].zfill(2)]
        if logx:
            xydata[:,0] = abs(xydata[:,0])
        if logy:
            xydata[:,1] = abs(xydata[:,1])
        xydata.sort(axis=0)
        xy1, xy2 = xydata
        xi = self.get_xi(xy1[0], xy2[0])
        func = openmc.data.Tabulated1D(*xydata.T, breakpoints=[2], interpolation=[scheme])
        if PLOT:
            x = np.linspace(xy1[0],xy2[0], 1000)
            plt.plot(x, func(x))
            plt.xticks([xy1[0], xi, xy2[0]])
            plt.show()
        numerical_result = quadrature(func, xy1[0], xi, rtol=1E-12)[0]
        analytical_result= area_between_2_pts(xy1, xy2, xi, scheme)
        # print(f"{numerical_result=}", f"{analytical_result=}", f"for {scheme=}")
        return np.isclose(numerical_result, analytical_result)

    def get_xi(self, xmin, xmax, full_step=False):
        if full_step:
            return xmax
        else:
            step_size = rn.uniform()*(xmax-xmin)
            return xmin + step_size

def create_func():
    interpolation = [1,2]
    x = [0,1,2, 3,4]
    y =  [1,2,3,1,3]
    breakpoints=[4,5] 
    interpolation=[1,2]
    return openmc.data.Tabulated1D(x, y, breakpoints, interpolation)

def func_area():
    return [1,2,3,(3-1)*(1)/2 +1]

class IntegratorTest(unittest.TestCase):
    def test_create_integral(self):
        integrated_func = Integrate(create_func())
        return integrated_func
    def test_check_integral(self):
        I = self.test_create_integral()
        self.assertEqual(func_area(), I.next_area)