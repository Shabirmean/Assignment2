'''
Copyright (c) 2017, Juan Camilo Gamboa Higuera, Anqi Xu, Victor Barbaros, Alex Chatron-Michaud, David Meger

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import math
from plant import gTrig_np
from cartpole import default_params
from cartpole import CartpoleDraw


# np.random.seed(31337)
np.set_printoptions(linewidth=500)


# FOR YOU TODO: Fill in this function with a control
# policy that computes a useful u from the input x
prev_angle_error = math.pi
# prev_cart_pos = 0.0

def policyfn(x):
    global prev_angle_error
    # global prev_cart_pos

    u = 0
    distance = x[0]
    cart_velocity = x[1]
    pendulum_angularV = x[2]
    sin_theta = x[3]
    cos_theta = x[4]

    theta_radians = math.acos(cos_theta)
    theta_degrees = math.degrees(theta_radians)

    pi = math.pi
    zero = 0.0
    p_tau = 0.2
    d_tau = 0.0
    i_tau = 0.01

    pendulum_position_error = pi - theta_radians
    if sin_theta < 0:
        pendulum_position_error = -pendulum_position_error

    pendulum_speed_error = zero - pendulum_angularV
    pendulum_state_error = pendulum_position_error + pendulum_speed_error

    cart_position_error = zero - distance
    cart_speed_error = zero - cart_velocity
    cart_state_error = cart_position_error + cart_speed_error

    # if sin_theta > 0:
    u_pendulum = (p_tau * pendulum_state_error) + (d_tau * pendulum_angularV)
    u_cart = (p_tau * cart_state_error) + (d_tau * cart_velocity)
    # else:
    #     u_pendulum = (-p_tau * pendulum_state_error) + (-d_tau * pendulum_angularV)
    #     u_cart = (-p_tau * cart_state_error) + (-d_tau * cart_velocity)

    u = u_cart - u_pendulum

    # velocity_error = 0 - pendulum_angularV
    # u = p_tau * (angle_error + velocity_error)

    # print ("X Distance: " + str(distance))
    # print ("Cart Velocity: " + str(cart_velocity))
    # print ("Angular Velocity: " + str(pendulum_angularV))
    # print ("Sin Theta: " + str(sin_theta))
    # print ("Cos Theta: " + str(math.degrees(math.acos(cos_theta))))
    # print ()

    return np.array([u])


def apply_controller(plant, params, H, policy=None):
    '''
    Starts the plant and applies the current policy to the plant for a duration specified by H (in seconds).

    @param plant is a class that controls our robot (simulation)
    @param params is a dictionary with some useful values 
    @param H Horizon for applying controller (in seconds)
    @param policy is a function pointer to the code that implements your 
            control solution. It will be called as u = policy( state )
    '''

    # start robot
    x_t, t = plant.get_plant_state()
    if plant.noise is not None:
        # randomize state
        Sx_t = np.zeros((x_t.shape[0], x_t.shape[0]))
        L_noise = np.linalg.cholesky(plant.noise)
        x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise)

    sum_of_error = 0
    H_steps = int(np.ceil(H / plant.dt))

    global prev_cart_pos
    prev_cart_pos = gTrig_np(x_t[None, :], params['angle_dims']).flatten()[0]

    for i in xrange(H_steps):
        # convert input angle dimensions to complex representation
        x_t_ = gTrig_np(x_t[None, :], params['angle_dims']).flatten()
        #  get command from policy (this should be fast, or at least account for delays in processing):
        u_t = policy(x_t_)

        #  send command to robot:
        plant.apply_control(u_t)
        plant.step()
        x_t, t = plant.get_plant_state()
        l = plant.params['l']
        err = np.array([0, l]) - np.array([x_t[0] + l * x_t_[3], -l * x_t_[4]])
        dist = np.dot(err, err)
        sum_of_error = sum_of_error + dist

        if plant.noise is not None:
            # randomize state
            x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise)

        if plant.done:
            break

    print "Error this episode %f" % (sum_of_error)

    # stop robot
    plant.stop()


def main():
    # learning iterations
    N = 5
    H = 10

    learner_params = default_params()
    plant_params = learner_params['params']['plant']
    plant = learner_params['plant_class'](**plant_params)

    draw_cp = CartpoleDraw(plant)
    draw_cp.start()

    # loop to run controller repeatedly
    for i in xrange(N):
        # execute it on the robot
        plant.reset_state()
        apply_controller(plant, learner_params['params'], H, policyfn)


if __name__ == '__main__':
    main()
