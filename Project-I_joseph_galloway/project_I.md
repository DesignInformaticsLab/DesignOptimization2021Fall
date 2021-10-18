# **Project I**

Author: Joseph D. Galloway II <br>
Course: Design Optimization (ME 598)


#Better Problem Formulation <br>
The following is included in the new problem formulation...
1) Rocket orientation (theta) and angular velocity (theta_dot)

So the new state space will be the following: <br>
d(t + 1) = d(t) + v(t)*delta_t <br>
v(t + 1) = v(t) + a(t)*delta(t) <br>
theta(t + 1) = theta(t) + theta_dot(t)*delta_t <br>
theta_dot(t + 1) = theta_dot(t) + theta_dotdot(t)*delta_t <br>

where delta_t is a time interval.<br>

The control input of the rocket is its acceleration, a(t), and the angular acceleration, theta_dotdot(t).

# Rocket Dimensions
When the angle of the rocket (theta) equals zero then the rocket is facing upright. Theta is positive when the angle is counter-clockwise from zero and negative when the angle is clockwise from zero.
