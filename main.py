import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegressionScratch

n_samples, n_features = 10, 1
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
