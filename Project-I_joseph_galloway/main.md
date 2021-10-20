---
author: Joseph D. Galloway II
date: October 20th, 2021
title: "Project I: Rocket Landing"
---

# Better Problem Formulation

Out of the options given, the following was chosen for the better
problem formulation:

1\) Rocket orientation ($\theta$) and angular velocity ($\dot{\theta}$)

The rocket state is represented by the rocket's distance to the ground,
d(t), its velocity, v(t), its orientation, $\theta$(t), and its angular
velocity, $\dot{\theta}$(t), i.e.
$x(t) = \lbrack d(t),v(t),\theta(t),\dot{\theta}(t)\rbrack^{T}$ where t
specifies time. The control input of the rocket is its acceleration a(t)
and angular acceleration $\ddot{\theta}$(t). The discrete-time dynamics
are the following:

$$d(t + 1) = d(t) + v(t)\Delta t$$

$$v(t + 1) = v(t) + a(t)\Delta t$$

$$\theta(t + 1) = \theta(t) + \dot{\theta}(t)\Delta t$$

$$\dot{\theta}(t + 1) = \dot{\theta}(t) + \ddot{\theta}(t)\Delta t$$

where $\Delta t$ is a time interval. Let the closed-loop controller be
the following:

$$a_{1}(t) = f_{\alpha}(x(t))$$

$$a_{2}(t) = f_{\alpha}(x(t))$$

where $f_{\alpha}(*)$ is a neural network with parameters $\alpha$,
which are to be determined through optimization. $a_{1}$(t) controls the
linear motion of the rocket while $a_{2}$(t) controls the rocket's
rotation.

For each time step, we assign a loss as a function of the control inputs
and the state: $l(x(t),a_{1}(t),a_{2}(t))$. Let
$l(x(t),a_{1}(t),a_{2}(t)) = 0$ for all t = 1, \..., T - 1, where T is
the final step, and
$l(x(T),a_{1}(T),a_{2}(T)) = ||x(T)||^{2} = d(T)^{2} + v(T)^{2} + \theta(T)^{2} + \dot{\theta}(T)^{2}$.
This loss encourages the rocket to reach $d(T) = 0$, $v(T) = 0$,
$\theta(T) = 0$, and $\dot{\theta}(T) = 0$, which are proper landing
conditions.

The optimization problem is now formulated as

$$min_{\alpha}\quad||x(T)||^{2}$$

$$d(t + 1) = d(t) + v(t)\Delta t$$

$$v(t + 1) = v(t) + a(t)\Delta t$$

$$\theta(t + 1) = \theta(t) + \dot{\theta}(t)\Delta t$$

$$\dot{\theta}(t + 1) = \dot{\theta}(t) + \ddot{\theta}(t)\Delta t$$

$$a_{1}(t) = f_{\alpha}(x(t)),\ \forall\quad t = 1,...,T - 1$$

$$a_{2}(t) = f_{\alpha}(x(t)),\ \forall\quad t = 1,...,T - 1$$

While this problem is constrained, it is easy to see that the objective
function can be expressed as a function of $x(T - 1)$ and $a_{1}(T - 1)$
and $a_{2}(T - 1)$. Thus it is essentially an unconstrained problem with
respect to $\alpha$.

The following assumptions are made:

1.  When $\theta = 0$, the rocket is oriented upright, which is the
    correct landing orientation.

2.  The rocket is upright when the axis along the height of the rocket
    is perpendicular to the ground. $\theta$ equals zero when the rocket
    is upright. $\theta$ is positive when the rocket rotates
    counter-clockwise from zero and negative when the rocket rotates
    clockwise from zero.

3.  The constants defined in the beginning of the code are all
    assumptions. It is assumed the gravity and rotation accelerations
    will always act on the rocket.

4.  The system was designed so that the rotation acceleration always
    acts on the rocket, making it turn counter clockwise. The side
    thruster/booster was designed to always act on the rocket, making it
    turn clockwise. It is assumed these acceleration directions will not
    change.

5.  The initial state of d(t) $>$ 0. The initial velocity and angular
    velocity equal zero. The initial angle of the rocket can be positive
    or negative.

6.  There are no other forces acting on the rocket such as drag (Option
    number 2).

7.  There are no other constraints in the state and action spaces
    (Option number 3)

8.  The controller is only designed for the initial states specified in
    the results. It is assumed the system cannot handle initial states
    other than the values specified in the code (Option number 4).

9.  There is no randomness in the dynamics or sensing of the rocket
    (Option number 5).

10. There is no discontinuity in modeling such as mechanical failures
    (Option number 6).

# Analysis of the Results

The initial states of the system are the following: $d(t) = 1$,
$v(t) = 0$, $\theta(t) = - 1$, and $\dot{\theta}(t) = 0$. These initial
states mean the following:

1.  $d(t) = 1$ means the rocket is initially at a distance of 1 above
    the ground.

2.  $v(t) = 0$ means the rocket is initially at the apex of its
    trajectory and is beginning its descent.

3.  $\theta(t) = - 1$ means the rocket is initially at angle of -1
    clockwise with respect to its upright axis.

4.  $\dot{\theta}(t) = 0$ means the rocket is initially not rotating.

The final results show that the rocket ended upright ($\theta(T) = 0$)
on the ground ($d(T) = 0$) with no linear velocity ($v(T) = 0$) and no
angular velocity ($\dot{\theta}(T) = 0$). These conditions are met
because the loss function equals zero.
$l(x(T),a_{1}(T),a_{2}(T)) = ||x(T)||^{2} = d(T)^{2} + v(T)^{2} + \theta(T)^{2} + \dot{\theta}(T)^{2} = 0$.
The optimizer reached these results in 40 iterations.

![Final iteration of the
system](media/rId21.png){width="4.536841644794401in"
height="1.9789468503937009in"}

Final iteration of the system
