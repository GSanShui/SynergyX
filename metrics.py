from math import sqrt
from scipy.stats import pearsonr
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import warnings
warnings.filterwarnings("ignore")

def mae(y, f):
    return mean_absolute_error(y, f)


def rmse(y, f):
    rmse = sqrt(mean_squared_error(y, f))
    return rmse


def mse(y, f):
    mse = mean_squared_error(y, f)
    return mse


def pearson(y, f):
    # rp = np.corrcoef(y, f)[0, 1]
    rp = pearsonr(y.flatten(), f.flatten())[0]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def r2(y, f):
    return r2_score(y, f)


def get_metrics(y,f):
    return mse(y, f), rmse(y, f), mae(y, f), r2(y, f), pearson(y, f), spearman(y, f)

