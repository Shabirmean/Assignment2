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


def getRadInTwoPi(sin_theta, cos_theta):
    if sin_theta < 0:
    	if cos_theta < 0:
    		angle = (math.pi) + -math.asin(sin_theta)
    	else:
    		angle = (2 * math.pi) + math.asin(sin_theta)
    else:
    	if cos_theta < 0:
    		angle = (math.pi) - math.asin(sin_theta)
    	else:
    		angle = math.asin(sin_theta)
    
    #print ("Sin_theta - " + str(sin_theta) + ", Angle - " + str(angle))
    return angle 


def getRadInTwoPi2(sin_theta, cos_theta):
    if sin_theta < 0:
	angle = math.asin(sin_theta)
    else:
	angle = (2 * math.pi) - math.asin(sin_theta)
    
    #print ("Sin_theta - " + str(sin_theta) + ", Angle - " + str(angle))
    return angle 


# FOR YOU TODO: Fill in this function with a control
# policy that computes a useful u from the input x
total_theta_error = 0.0
total_distance_error = 0.0
prev_theta_error_sign = 1
prev_distance_error_sign = 1

def policyfn(x, p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
#def policyfn(x):
    global total_theta_error
    global total_distance_error
    global prev_theta_error_sign
    global prev_distance_error_sign

    u = 0
    distance = x[0]
    cart_velocity = x[1]
    pendulum_angularV = x[2]
    sin_theta = x[3]
    cos_theta = x[4]

    #theta_radians = math.acos(cos_theta)
    #theta_degrees = math.degrees(theta_radians)

    pi = math.pi
    zero = 0.0

    #Kp_cart = 50.0    #0.7
    #Kd_cart = 0.0    #0.2
    #Ki_cart = 0.0
    #Kp_pendulum = 500.0
    #Kd_pendulum = 0.0
    #Ki_pendulum = 0.0

    Kp_cart = p[3]    #0.7
    Kd_cart = p[4]    #0.2
    Ki_cart = p[5]
    Kp_pendulum = p[0]
    Kd_pendulum = p[1]
    Ki_pendulum = p[2]

    cart_position_error = zero - distance
    cart_speed_error = zero - cart_velocity
    #cart_speed_error = -cart_velocity
    
    angle = getRadInTwoPi(sin_theta, cos_theta)
    pendulum_position_error = pi - angle
    pendulum_speed_error = zero - pendulum_angularV
    #pendulum_speed_error = -pendulum_angularV

#    if angle > (pi / 2.0) and angle < (3*pi / 2.0):
# 	pendulum_position_error = pi - angle
#    elif angle < (pi / 2.0):	
#	small_angle = (pi / 2.0) - angle
#	pendulum_position_error = pi - small_angle
#    elif angle > (3*pi / 2.0):
#	small_angle = angle - (3*pi / 2.0)
# 	small_angle = (2 *  pi) - small_angle
#	pendulum_position_error = pi - small_angle
    
    if (pendulum_position_error * prev_theta_error_sign > 0):
        total_theta_error += pendulum_position_error
    else:
        total_theta_error = pendulum_position_error
        prev_theta_error_sign *= -1 
    
    if (cart_position_error * prev_distance_error_sign > 0):
        total_distance_error += cart_position_error
    else:
        total_distance_error = cart_position_error
        prev_distance_error_sign *= -1
    
    
    u_cart = (Kp_cart * cart_position_error) + (Kd_cart * cart_speed_error) + (Ki_cart * total_distance_error)
    u_pendulum = (Kp_pendulum * pendulum_position_error) + \
                    (-Kd_pendulum * pendulum_speed_error) + \
                            (Ki_pendulum * total_theta_error)
    
    #u = (u_cart + u_pendulum)
    u = u_pendulum - u_cart
    #print ("theta_error: " + str(pendulum_position_error) + ", u_pendulum: " + \
     #   str(u_pendulum) + ", u: " + str(u))

    return np.array([u])


#def apply_controller(plant, params, H, policy=None):
def apply_controller(plant, params, H, best_err, p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], policy=None):
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
    prev_error = 0
    H_steps = int(np.ceil(H / plant.dt))


    for i in xrange(H_steps):
        #print("Step: "+ str(i) + " of " + str(H_steps))
        # convert input angle dimensions to complex representation
        x_t_ = gTrig_np(x_t[None, :], params['angle_dims']).flatten()
        #  get command from policy (this should be fast, or at least account for delays in processing):
        u_t = policy(x_t_, p)

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

        if sum_of_error > best_err:
            if not i == 0 and prev_error < dist:
                break

        prev_error = dist

        if plant.done:
            break

    print "Error this episode %f" % (sum_of_error)

    # stop robot
    plant.stop()
    return sum_of_error


def twiddle(plant, learner_params, H, tol=0.2): 
    #[99.77458510000008, -0.8421347170045719, 0.0]
    #p = [100, 50, 0, 0, 0, 0]
    #dp = [1, 1, 1, 1, 1, 1]
    #p = [99.77458510000008, -0.8421347170045719, 0]
    p = [100.02169285205473, -0.9511347170045724, 0.0, 50.0, 0.0, 0.0]
    dp = [1, 1, 1, 1, 1, 1]
    plant.reset_state()

    best_err = 10000000.0
    best_err = apply_controller(plant, learner_params['params'], H, best_err, p, policyfn)

    it = 0
    while sum(dp) > tol:
        print("Iteration {}, best error = {}".format(it, best_err))
        for i in range(3, len(p)):
            p[i] += dp[i]
            plant.reset_state()
            err = apply_controller(plant, learner_params['params'], H, best_err, p, policyfn)

            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] -= 2 * dp[i]
                plant.reset_state()
                err = apply_controller(plant, learner_params['params'], H, best_err, p, policyfn)

                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p[i] += dp[i]
                    dp[i] *= 0.9
            print ("Controller was - " + str(i) + ", sum = " + str(sum(dp)) + ", " + str(dp))

        it += 1
    print ("dp-" + str(dp))
    return p


def main():
    # learning iterations
    N = 5
    H = 10

    learner_params = default_params()
    plant_params = learner_params['params']['plant']
    plant = learner_params['plant_class'](**plant_params)

    draw_cp = CartpoleDraw(plant)
    draw_cp.start()
    
    #params = twiddle(plant, learner_params, H, tol=3.1)
    #print("===== DONE =====")
    #print(params)

    # loop to run controller repeatedly
    p=[100.02169285205473, -0.9511347170045724, 0.0, 50.77308999999996, 2.1445497382874485, 0.0]
    best_err = 10000000.0
    for i in xrange(N):
        # execute it on the robot
        plant.reset_state()
        apply_controller(plant, learner_params['params'], H, best_err, p, policyfn)


if __name__ == '__main__':
    main()
