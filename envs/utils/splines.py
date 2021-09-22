import numpy as np

import bezier

class BezierGeneratorCartesian():

    def __init__(self,
                 duration=5000,
                 n_samples=5,
                 origin=[0.53, 0.00, 0.55],
                 width=1,
                 length=1,
                 height=1):
        self.n_samples = n_samples
        self.height = height
        self.origin = origin - np.array([length,width,height])/2 # center the sample space
        self.width = width
        self.length = length
        self.duration = duration
        self.generate_new_curve()

    def evaluate(self,t):
        t = t/self.duration
        if t > 1.0:
            t = 1.0
        point = np.squeeze(self.curve.evaluate(float(t)))
        orientation = np.array([0, -np.pi, 0])
        return np.concatenate((point, orientation))

    def generate_new_curve(self):
        nodes = np.random.random_sample((self.n_samples, 3))
        nodes[:,0] *= self.length
        nodes[:,1] *= self.width
        nodes[:,2] *= self.height
        nodes+=self.origin
        nodes_ft = np.asfortranarray(nodes.transpose())
        self.curve = bezier.Curve(nodes_ft, degree=int(self.n_samples-1))

class BezierGeneratorSpherical():
    # defaults take from
    # https://www.making.unsw.edu.au/media/images/Spez_LBR_iiwa_en_zaP9fSa.original.jpg

    def __init__(self,
                 duration=5000,
                 n_samples=5,
                 r_range = [0.420,0.820],
                 theta_range = [-1.25*np.pi,1.25*np.pi],
                 phi_range = [-2.0944,2.0944],
                 origin = [0,0,0.360]
                 ):
        self.n_samples = n_samples
        self.r_range = r_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.duration = duration
        self.origin = origin
        self.generate_new_curve()

    def evaluate(self,t):
        t = t/self.duration
        if t > 1.0:
            t = 1.0
        point_spherical = np.squeeze(self.curve.evaluate(float(t)))
        point = self.sphere2cart(point_spherical)
        point += self.origin
        orientation = np.array([0, -np.pi, 0])
        return np.concatenate((point, orientation))

    def generate_new_curve(self):
        nodes = np.random.random_sample((self.n_samples, 3))
        nodes[:,0] = nodes[:,0]*(self.r_range[1]-self.r_range[0]) + self.r_range[0]
        nodes[:,1] = nodes[:,1]*(self.theta_range[1]-self.theta_range[0]) + self.theta_range[0]
        nodes[:,2] = nodes[:,2]*(self.phi_range[1]-self.phi_range[0]) + self.phi_range[0]
        nodes_ft = np.asfortranarray(nodes.transpose())
        self.curve = bezier.Curve(nodes_ft, degree=int(self.n_samples-1))

    def sphere2cart(self, point):
        r = point[0]
        theta = point[1]
        phi = point[2]
        return np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

def baseline_circle_trajectory(t):
    # TODO make this editable online
    radius = 0.1
    [0.537, 0.0, 0.5]
    x = 0.5321173259355029
    y = -0.0011066349287836563
    z = 0.55229843834858136
    x += 0
    y += np.sin(2*t*2*np.pi/5000)*radius
    z += (np.cos(2*t*2*np.pi/5000)-1)*radius
    return np.array([x, y, z, 0, -np.pi, 0])



def trajectory_pol(xi,xti,xtti,xf,xtf,xttf,T):
    return lambda t:(6*t**5*xf)/T**5 - (15*t**4*xf)/T**4 + (10*t**3*xf)/T**3 + xi - (6*t**5*xi)/T**5 + (15*t**4*xi)/T**4 - (10*t**3*xi)/T**3 - (3*t**5*xtf)/T**4 + (7*t**4*xtf)/T**3 - (4*t**3*xtf)/T**2 + t*xti - (3*t**5*xti)/T**4 + (8*t**4*xti)/T**3 - (6*t**3*xti)/T**2 + (t**5*xttf)/(2.*T**3) - (t**4*xttf)/T**2 + (t**3*xttf)/(2.*T) + (t**2*xtti)/2. - (t**5*xtti)/(2.*T**3) + (3*t**4*xtti)/(2.*T**2) - (3*t**3*xtti)/(2.*T)
