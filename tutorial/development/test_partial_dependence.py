import sys

dirname = "/Users/m.daffarobani/Documents/personal_research/smt"
if dirname not in sys.path:
    sys.path.append(dirname)


from smt.problems import Rosenbrock
from smt.sampling_methods import LHS

# to ignore warning messages
import warnings

warnings.filterwarnings("ignore")


plot_status = True

from smt_explainability import partial_dependence

ndim = 2
ndoe = 20  # int(10*ndim)
fun = Rosenbrock(ndim=ndim)
sampling = LHS(xlimits=fun.xlimits, criterion="ese", seed=1)
xt = sampling(ndoe)

features = [0]
model = None
pd_results = partial_dependence(model, xt, features)
