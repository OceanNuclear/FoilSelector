from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
import unittest
from flux_convert import flux_conversion
from misc_library import Integrate, MeV
import numpy.random as rn
from scipy.integrate import quadrature # get gaussian quadrature
import openmc 
from numpy import log as ln

class AreaTest(unittest.TestCase):
    '''
    def __init__(self):
        super.__init__(self)
        self.a=1
    '''
    # doesn't need an init method
    '''
    def setUp(): # overwrites the default
        with open("somefile.txt", "w") as f:
            self.data=f.write("words")

    def tearDown():
        os.remove("somefile.txt")
    '''
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

    def loop_through_scheme():
        for i in range(1,6):
            assert self.inbetween(i)
            assert self.fullrange(i)

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
    # def test_ln(self):
    #     from numpy import log as ln
    #     #make sure ln(y2/y1)/ln(x2/x1) = -1
    #     f = openmc.data.Tabulated1D()        

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
        integrated_func = Integrate2(create_func())
        return integrated_func
        
    def test_check_integral(self):
        I = self.test_create_integral()
        self.assertListEqual(func_area(), I._area)

    def test_dy_dx_appraoch_0(self):
        """
        Test that everything still works when dy -> 0 an dx -> 0
        """
        I = self.test_create_integral()
        # testing all dx=0 cases
        for i in openmc.data.INTERPOLATION_SCHEME.keys():
            self.assertEqual(getattr(I, "area_scheme_"+str(i))(1,1, 2,2), 0, "Make sure area=0 when x1!=0, y1!=0, dx=0, dy =0 for scheme={}".format(i))
            self.assertEqual(getattr(I, "area_scheme_"+str(i))(1,1, 2,3), 0, "Make sure area=0 when x1!=0, y1!=0, dx=0, dy!=0 for scheme={}".format(i))
            self.assertEqual(getattr(I, "area_scheme_"+str(i))(1,1, 0,0), 0, "Make sure area=0 when x1!=0, y1 =0, dx=0, dy =0 for scheme={}".format(i))
            self.assertEqual(getattr(I, "area_scheme_"+str(i))(1,1, 0,1), 0, "Make sure area=0 when x1!=0, y1 =0, dx=0, dy!=0 for scheme={}".format(i))

            self.assertEqual(getattr(I, "area_scheme_"+str(i))(0,0, 2,2), 0, "Make sure area=0 when x1 =0, y1!=0, dx=0, dy =0 for scheme={}".format(i))
            self.assertEqual(getattr(I, "area_scheme_"+str(i))(0,0, 2,3), 0, "Make sure area=0 when x1 =0, y1!=0, dx=0, dy!=0 for scheme={}".format(i))
            self.assertEqual(getattr(I, "area_scheme_"+str(i))(0,0, 0,0), 0, "Make sure area=0 when x1 =0, y1 =0, dx=0, dy =0 for scheme={}".format(i))
            self.assertEqual(getattr(I, "area_scheme_"+str(i))(0,0, 0,1), 0, "Make sure area=0 when x1 =0, y1 =0, dx=0, dy!=0 for scheme={}".format(i))
        # testing the very specific case of slope = -1 when test = 1.
        self.assertEqual(I.area_scheme_5(1,2, 2,1), 1*1*(ln(2)-ln(1)), "Scheme 5 should still work in the slope = -1 case.")
        # testing the dy=0, dx!=0 cases # algebraically only an issue in scheme 4 and 5.
        self.assertEqual(I.area_scheme_4(1,2, 1,1), 1, "Scheme 4 should still work when dy=0, y1!=0")
        self.assertEqual(I.area_scheme_4(1,2, 0,0), 0, "Scheme 4 should still work when dy=0, y1 =0")
        self.assertEqual(I.area_scheme_5(1,2, 1,1), 1, "Scheme 5 should still work when dy=0, y1!=0")
        self.assertEqual(I.area_scheme_5(1,2, 0,0), 0, "Scheme 5 should still work when dy=0, y1 =0")

    def test_accuracy(self):
        return np.isclose

class FluxConversionTest(unittest.TestCase):
    def test_in(self):
        self.gen_a_flux()
        self.gen_gs()
        #let's say the current flux was given per MeV
        new_flux = flux_conversion(self.flux, self.gs, 'per MeV', 'per eV')
        # integrate_new_flux = new_flux*self.gs_interval
        # integrate_old_flux = self.flux*self.gs_interval/1E6 
        # np.testing.assert_array_almost_equal_nulp(integrate_new_flux, integrate_old_flux)

    def test_out(self):
        pass
        
    def gen_a_flux(self):
        self.flux= rn.uniform(size=10)
    def gen_gs(self):
        self.gs = np.cumsum(rn.uniform(size=11))
        self.gs_interval = np.diff(self.gs)

def area_between_2_pts(xy1,xy2, xi,scheme):
    x1, y1, x2, y2, xi = np.hstack([xy1, xy2, xi]).flatten()
    assert x1<=xi<=x2, "xi must be between x1 and x2"
    x_ = xi-x1
    if x1==x2:
        return 0.0  #catch all cases with zero-size width bins
    if y1==y2 or scheme==1:# histogramic/ flat interpolation
        return y1*x_ #will cause problems in all log(y) interpolation schemes if not caught
    
    dy, dx = y2-y1, x2-x1
    logy, logx = [bool(int(i)) for i in bin(scheme-2)[2:].zfill(2)]
    if logx:
        assert(all(ary([x1,x2,xi])>0)), "Must use non-zero x values for these interpolation schemes"
        lnx1, lnx2, lnxi = ln(x1), ln(x2), ln(xi)
        dlnx = lnx2 - lnx1
    if logy:
        assert(all(ary([y1,y2])>0)), "Must use non-zero y values for these interpolation schemes"
        lny1, lny2 = ln(y1), ln(y2)
        dlny = lny2 - lny1

    if scheme==2: # linx, liny
        m = dy/dx
        if xi==x2:
            return dy*dx/2 + y1*dx
        else:
            return y1*x_ + m*x_**2 /2
    elif scheme==3: # logx, liny
        m = dy/dlnx
        if xi==x2:
            return y1*dx + m*(x2*dlnx-dx)
        else:
            return y1*x_ + m*(xi*(lnxi-lnx1)- x_)
            return (y1 - m*lnx1)*x_ + m*(-x_+xi*lnxi-x1*lnx1)
    elif scheme==4: # linx, logy
        m = dlny/dx
        if xi==x2:
            return 1/m *dy
        else:
            return 1/m *y1*(exp(x_*m)-1)
    elif scheme==5:
        m = dlny/dlnx
        if m==-1:
            return y1 * x1 * (lnxi-lnx1)
        if xi==x2:
            return y1/(m+1) * ( x2 * (x2/x1)**m - x1 )
        else:
            return y1/(m+1) * ( xi * (1 + x_/x1)**m - x1 )
    else:
        raise AssertionError("a wrong interpolation scheme {0} is provided".format(scheme))