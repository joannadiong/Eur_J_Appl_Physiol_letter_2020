import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import sympy


def fit(freq, angle, torque):
    # find min, max values in angle array
    angle_min, angle_max = np.min(angle), np.max(angle)
    # calculate fitted angle values based on sampling rate
    angle_fit = np.arange(angle_min, angle_max, 1 / freq)
    torque_fit = []
    # define exponential function
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    # initialise starting values: these take some fiddling around
    p0 = 5, 0.01, -10
    # fit exponential function to data, return optimal values and estimated covariance for parameters
    popt, pcov = curve_fit(func, angle, torque, p0, maxfev=10000)
    a, b, c = popt
    # calculate predicted torque using coefficients and angle
    for angle in angle_fit:
        y = a * np.exp(-b * angle) + c
        torque_fit.append(y)
    torque_fit = np.array(torque_fit)
    # return indices to sort array, then sort angle and torque
    sort_order = angle_fit.argsort()
    # find predicted angle at known torque
    angle_fit = angle_fit[sort_order]
    torque_fit = torque_fit[sort_order]
    # return fitted torque and angle, and predicted angle
    return angle_fit, torque_fit, a, b, c

# read data, initialise variables
df = pd.read_csv('torque_angle.csv')
freq = 2000

# fit torque-angle data with exponential, plot
angle, torque, a, b, c = fit(freq, df.angle, df.torque)

# get stiffness derivatives from fitted data
Angle1 = angle[int(0.6 * len(angle))]
Torque1 = torque[int(0.6 * len(torque))]
Angle2 = angle[int(0.9 * len(angle))]
Torque2 = torque[int(0.9 * len(torque))]

x = sympy.Symbol('x')
y = a * sympy.exp(-b * x) + c
ydiff = y.diff(x)

Angle = Angle1
y.subs(x, Angle)
ydiff.subs(x, Angle)
print('angle = {:.04} deg, torque = {:.04} Nm, stiffness = {:.04} Nm deg^-1'.format(Angle, y.subs(x, Angle), ydiff.subs(x, Angle)))
Angle = Angle2
y.subs(x, Angle)
ydiff.subs(x, Angle)
print('angle = {:.04} deg, torque = {:.04} Nm, stiffness = {:.04} Nm deg^-1'.format(Angle, y.subs(x, Angle), ydiff.subs(x, Angle)))
deltaAngle = Angle2 - Angle1
deltaTorque = Torque2 - Torque1
delta = deltaTorque / deltaAngle
print('wrong stiffness = {:.04} Nm deg^-1'.format(delta))

# plot figure
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(df.angle, df.torque, '0.5', label='raw data')
plt.plot(angle, torque, 'k', label='fitted curve')
plt.plot(Angle1, Torque1, 'ko')
plt.plot(Angle2, Torque2, 'ko')
plt.plot([Angle1, Angle2], [Torque1, Torque2], 'g--')
# Fig 1 red and blue dashed lines are added manually in Inkscape

plt.ylabel('Torque (Nm)')
plt.xlabel('Angle (deg)')
plt.legend()
plt.savefig('Fig1.png', dpi=300)