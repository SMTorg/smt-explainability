import sys

dirname = "/Users/m.daffarobani/Documents/personal_research/smt"
if dirname not in sys.path:
    sys.path.append(dirname)

from scipy import linalg
from smt.utils.misc import compute_rms_error

from smt.problems import Sphere, NdimRobotArm, Rosenbrock
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP

# to ignore warning messages
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import scipy.interpolate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

plot_status = True

from smt.explainability_tools._partial_dependence import partial_dependence

ndim = 2
ndoe = 20  # int(10*ndim)
fun = Rosenbrock(ndim=ndim)
sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
xt = sampling(ndoe)

features = [0]
model = None
pd_results = partial_dependence(model, xt, features)
