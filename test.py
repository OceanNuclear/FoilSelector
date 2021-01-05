from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
import unittest
from flux_convert import flux_conversion
from misc_library import Integrate, MeV
import numpy.random as nprn
import random as rn
from scipy.integrate import quadrature # get gaussian quadrature
import openmc 
from numpy import log as ln
from scipy.integrate import quad, quadrature, trapz
from misc_library import openmc_variable
vars().update(openmc_variable)

def slow_integrate(func, a, b, num=5000):
    """ makes no assumption about the function other than the fact that it's continuous. Works much better.
    """
    diff = b-a 
    x = np.linspace(a, b-diff*1E-9, num)
    y = func(x)
    return trapz(y, x=x)

def create_test_func():
    interpolation = [1,2]
    x = [0,1,2, 3,4]
    y =  [1,2,3,1,3]
    breakpoints=[4,5] 
    interpolation=[1,2]
    return openmc.data.Tabulated1D(x, y, breakpoints, interpolation)

def test_func_area():
    return [1,2,3,(3-1)*(1)/2 +1]

def create_random_func(n=20):
    # Create random data for x, y, and the interpolation scheme
    x = np.cumsum(nprn.rand(n+1)) # linearly increasing list of values
    y = 1-nprn.rand(n+1) # a list of all positive numbers, non-zero
    interpolation_long = rn.choices(list(openmc.data.INTERPOLATION_SCHEME.keys()), k=n+1)
    interpolation_long[-1] = 0
    breakpoints = np.sort(np.argwhere(np.diff(interpolation_long)).flatten()+2)
    interpolation = ary(interpolation_long)[breakpoints-2]

    tab = openmc.data.Tabulated1D(x, y, interpolation=interpolation, breakpoints=breakpoints)
    return tab

class TestIntegratorEdgeCases(unittest.TestCase):
    """
    Ensure that that there are no problem handling dx=0, dy=0, etc.
    """
    def test_dy_dx_appraoch_0_scalar(self):
        """
        Test that everything still works when dy -> 0 an dx -> 0
        """
        I = Integrate
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

    def test_dy_dx_appraoch_0_array(self):
        I = Integrate
        for i in openmc.data.INTERPOLATION_SCHEME.keys():
            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([1]),ary([1]), ary([2]),ary([2])).tolist(), [0], "Make sure area=0 when x1!=0, y1!=0, dx=0, dy =0 for scheme={}".format(i))
            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([1]),ary([1]), ary([2]),ary([3])).tolist(), [0], "Make sure area=0 when x1!=0, y1!=0, dx=0, dy!=0 for scheme={}".format(i))
            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([1]),ary([1]), ary([0]),ary([0])).tolist(), [0], "Make sure area=0 when x1!=0, y1 =0, dx=0, dy =0 for scheme={}".format(i))
            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([1]),ary([1]), ary([0]),ary([1])).tolist(), [0], "Make sure area=0 when x1!=0, y1 =0, dx=0, dy!=0 for scheme={}".format(i))

            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([0]),ary([0]), ary([2]),ary([2])).tolist(), [0], "Make sure area=0 when x1 =0, y1!=0, dx=0, dy =0 for scheme={}".format(i))
            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([0]),ary([0]), ary([2]),ary([3])).tolist(), [0], "Make sure area=0 when x1 =0, y1!=0, dx=0, dy!=0 for scheme={}".format(i))
            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([0]),ary([0]), ary([0]),ary([0])).tolist(), [0], "Make sure area=0 when x1 =0, y1 =0, dx=0, dy =0 for scheme={}".format(i))
            self.assertListEqual(getattr(I, "area_scheme_"+str(i))(ary([0]),ary([0]), ary([0]),ary([1])).tolist(), [0], "Make sure area=0 when x1 =0, y1 =0, dx=0, dy!=0 for scheme={}".format(i))
        # testing the very specific case of slope = -1 when test = 1.
        self.assertListEqual(I.area_scheme_5(ary([1]),ary([2]), ary([2]),ary([1])).tolist(), [1*1*(ln(2)-ln(1))], "Scheme 5 should still work in the slope = -1 case.")
        # testing the dy=0, dx!=0 cases # algebraically only an issue in scheme 4 and 5.
        self.assertListEqual(I.area_scheme_4(ary([1]),ary([2]), ary([1]),ary([1])).tolist(), [1], "Scheme 4 should still work when dy=0, y1!=0")
        self.assertListEqual(I.area_scheme_4(ary([1]),ary([2]), ary([0]),ary([0])).tolist(), [0], "Scheme 4 should still work when dy=0, y1 =0")
        self.assertListEqual(I.area_scheme_5(ary([1]),ary([2]), ary([1]),ary([1])).tolist(), [1], "Scheme 5 should still work when dy=0, y1!=0")
        self.assertListEqual(I.area_scheme_5(ary([1]),ary([2]), ary([0]),ary([0])).tolist(), [0], "Scheme 5 should still work when dy=0, y1 =0")            

class TestIntegratorAccuracy(unittest.TestCase):
    """Ensure that the accuracy is within acceptable limit.
    """
    def create_known_integral(self):
        func = create_test_func()
        return func, Integrate(func)

    def create_random_integral(self, n=20):
        func = create_random_func(n)
        return func, Integrate(func)

    def test_check_integral(self):
        func, I = self.create_known_integral()
        self.assertListEqual(test_func_area(), I._area.tolist())

    def test_area_accuracy(self):
        func_ana, I_analytical = self.create_known_integral()
        func_random, I_random = self.create_random_integral()
        # wrapped_func_scalar = lambda x: func_random([x])[0]

        self.assertListEqual(I_analytical._area.tolist(), test_func_area(), "The analytically calculated area does not fit the expect answer")

        for ind, (x_low, x_upp) in enumerate(zip(func_random.x[:-1], func_random.x[1:])):
            self.assertAlmostEqual(I_random._area[ind], slow_integrate(func_random, x_low, x_upp), places=5, msg="the {}-th cell area is inaccurately calculated using scheme {}!".format(ind, I_random._interpolation[ind]))
            # self.assertAlmostEqual(I_random._area[ind], slow_integrate(wrapped_func_scalar, x_low, x_upp), msg="the {}-th cell area is inaccurately calculated!".format(ind))
            self.break_func1 = func_random, I_random

    def test_random_intervals(self):
        func_ana, I_analytical = self.create_known_integral()
        func_random, I_random = self.create_random_integral()
        # wrapped_func_scalar = lambda x: func_random([x])[0]

        # test array approach
        # get a few random upper and lower limits
        for range_x_low, range_x_upp in np.clip(np.cumsum(nprn.rand(8).reshape([-1,2]), axis=-1), 0, 1): # create four pairs
            upper = func_ana.x.min() + range_x_upp*(func_ana.x.max() - func_ana.x.min())
            lower = func_ana.x.min() + range_x_low*(func_ana.x.max() - func_ana.x.min())
            self.assertAlmostEqual(I_analytical.definite_integral(lower, upper), slow_integrate(func_ana, lower, upper), msg="The analytically constructed function doesn't integrate properly using the formula given!")
            self.break_func2 = func_ana, I_analytical

            upper = func_random.x.min() + range_x_upp*(func_random.x.max() - func_random.x.min())
            lower = func_random.x.min() + range_x_low*(func_random.x.max() - func_random.x.min())
            self.assertAlmostEqual(I_random.definite_integral(lower, upper), slow_integrate(func_random, lower, upper), msg="The randomly constructed function doesn't integrate properly using the formula given!")
            self.break_func3 = func_random, I_random
            # self.assertAlmostEqual(I_random.definite_integral(lower, upper), slow_integrate(wrapped_func_scalar, lower, upper), "The definite integral of the randomly constructed function doesn't give the same result as if we integrate the wrapped function!")

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
        self.flux= nprn.uniform(size=10)
    def gen_gs(self):
        self.gs = np.cumsum(nprn.uniform(size=11))
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

if __name__=='__main__':
    unittest.main()