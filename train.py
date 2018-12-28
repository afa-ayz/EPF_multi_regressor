


import os
from utils import *


if __name__ == "__main__":

    # Load data
    csv_list = ["Data/" + x for x in os.listdir("Data/") if 'csv' in x]

    temp_holder = []
    for each_csv in csv_list:
        temp_holder.append(load_data(each_csv))

    raw_data = np.concatenate(temp_holder, axis=0)
    data, target = raw_data[:, :-1], raw_data[:, -1:]


    # Pre-processing
    from sklearn.preprocessing import scale, minmax_scale, maxabs_scale, normalize, PolynomialFeatures
    data = scale(data) # Sample usage


    # Split data
    ratio = 0.1 # Sample usage
    idx_test = np.random.choice(len(data), int(ratio * len(data)), replace=False)
    idx_train = np.setxor1d(np.arange(len(data)), idx_test)

    train_data, train_target = data[idx_train], target[idx_train]
    test_data, test_target = data[idx_test], target[idx_test]


    # Generate random targets as baseline
    random_target = np.random.uniform(np.percentile(target, 5), np.percentile(target, 95), test_target.shape)



    # Load model
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, \
    LassoLarsIC, ElasticNet, ElasticNetCV, Lars, LassoLars, BayesianRidge, ARDRegression,  \
    SGDRegressor, RANSACRegressor, HuberRegressor

    from sklearn.svm import SVR, LinearSVR, NuSVR
    from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor

    model = KNeighborsRegressor() # Sample usage. Note that the parameters of each model can be fine-tuned for better performance


    # Train
    model.fit(train_data, train_target)


    # Evaluate
    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


    print(mean_absolute_error(test_target, model.predict(test_data))) # Sample usage
    print(mean_absolute_error(test_target, random_target))


