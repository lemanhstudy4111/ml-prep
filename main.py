import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegressionScratch
"""
   x1  x2
   1   2
   2   3
   
w1= 2, w2 = 3
2*1 + 3*2 = 8
2*2 + 3*3 = 13
"""
print(np.dot(np.array([[1,2], [2,3]]), np.array([2,3])))

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
# Always scale the input. The most convenient way is to use a pipeline.
reg = make_pipeline(StandardScaler(), SGDRegressor(penalty=None))
reg.fit(X, y)
randoms_X = np.random.rand()


reg_scratch = LinearRegressionScratch()
reg_scratch.fit(X, y)

sklearn_y = reg.predict([randoms_X])
scratch_y = reg_scratch.predict([randoms_X])
print(f"scikit-learn {sklearn_y}, scratch {scratch_y}")
