import numpy as np
import math
import gym

class Environment(object):
    """
    generic class for environments
    """
    def reset(self):
        """
        returns initial observation
        """
        pass

    def step(self, action):
        """
        returns (observation, termination signal)
        """
        pass


class CartpoleEnv(Environment):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
        The pole either starts in an upright position or a downward position. 
        The goal is to keep the pole upright within some thresholds by applying force to the cart. 
        
        Modified from the cartpole environment on OpenAI gym. 

    Observation: 
        0   Cart Position
        1   Cart Velocity
        2   Pole Angle
        3   Pole Velocity At Tip
        
    Actions:
        0   Push cart to the left
        1   No action
        2   Push cart to the right

    Starting State:
        Cart position, velocity, and angular velocity are drawn uniformly from [-0.05, 0.05]. 
        Pole angle is drawn uniformly from [-0.05, 0.05] if starting upright, and from 
        [pi-0.05, pi+0.05] if starting downwards. 

    Episode Termination:
        Cart Position is more than some threshold away from 0. 
    """

    def __init__(self, swing_up=False, timescale=0.02):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = timescale  # seconds between state updates
        self.state = None #(x, x_dot, theta, theta_dot)
        self.x_threshold = 5
        self.swing_up = swing_up # determines pole's initial position

    def step(self, action):
        assert action in [0, 1, 2], "invalid action"
        x, x_dot, theta, theta_dot = self.state
        force = (action - 1) * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta = np.remainder(theta, 2*math.pi)
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        done = bool((x<-self.x_threshold) or (x>self.x_threshold))
        
        return np.array(self.state), done

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.swing_up:
            self.state[2] += math.pi

        return np.array(self.state)

## test
if __name__ == '__main__':
    nsteps = 200

    np.random.seed(0)
    env = gym.make('MountainCar-v0') 
    obs = env.reset()
    t = 0
    print("t=%d, x %.2f, x_dot %.2f" % (t, obs[0], obs[1]))

    done = False
    while not done:
        env.render()
        action = np.random.randint(3)
        obs, reward, done, info= env.step(action)
        t += 1
        print("t=%d, action %d, x %.2f, x_dot %.2f, done %r" 
            %(t, action, obs[0], obs[1], done))
        done = done or t == nsteps
    env.render()