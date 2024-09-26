import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(123)

x_grid = np.linspace(0, 10, 100)

def true_fn(x):
    noises = np.random.normal(loc=0, scale=0.3**0.5, size=len(x))
    return np.array([ele+math.sin(1.5*ele) for ele in x])+noises

def fx(x):
    return np.array([ele+math.sin(1.5*ele) for ele in x])

# Overall code & procedure was followed by https://dustinstansbury.github.io/theclevermachine/bias-variance-tradeoff as assignment suggested.
# Q 1-2.
sample_num=20
# Sample data & sort
xs = np.random.rand(20)*10
xs.sort()
plt.plot(x_grid, fx(x_grid), label='f(x)', color='red')
plt.scatter(xs, true_fn(xs), label='y')
plt.title("Q1.2")
plt.legend()
plt.savefig("Q1_2.png")
plt.clf()
# Q 1-3.
degrees = [1, 3, 10]
colors = ['orange', 'royalblue', 'darkgreen']
theta = {}
fit = {}
plt.plot(x_grid, fx(x_grid), label='f(x)', color='red')
plt.scatter(xs, true_fn(xs), label='y')
for ii, degree in enumerate(degrees):
    # Note: we should get an overconditioned warning for degree 10 because of extreme overfitting
    theta[degree] = np.polyfit(xs, true_fn(xs), degree)
    fit[degree] = np.polyval(theta[degree], x_grid)
    plt.plot(x_grid, fit[degree], colors[ii], label=f"Degree={degree}")
plt.legend()
plt.ylim(0, 10)
plt.title("Q1.3")
plt.savefig("Q1_3.png")

# Q - 1.4
np.random.seed(42)
size_per_dataset = 50
n_datasets = 100
max_poly_degree = 15  # Maximum model complexity
model_poly_degrees = range(1, max_poly_degree + 1)

n_train = 40

# Create training/testing inputs
x = np.linspace(0, 10, size_per_dataset)
x = np.random.permutation(x)
x_train = x[:n_train]
x_test = x[n_train:]

# logging variables
theta_hat = defaultdict(list)

pred_train = defaultdict(list)
pred_test = defaultdict(list)

train_errors = defaultdict(list)
test_errors = defaultdict(list)

def error_function(pred, label):
    return (label-pred)**2

# Loop over datasets
for dataset in range(n_datasets):

    # Simulate training/testing targets
    y_train = true_fn(x_train)
    y_test = true_fn(x_test)
    # Loop over model complexities
    for degree in model_poly_degrees:
        # Train model
        tmp_theta_hat = np.polyfit(x_train, y_train, degree)

        # Make predictions on train set
        tmp_pred_train = np.polyval(tmp_theta_hat, x_train)
        pred_train[degree].append(tmp_pred_train)

        # Test predictions
        tmp_pred_test = np.polyval(tmp_theta_hat, x_test)
        pred_test[degree].append(tmp_pred_test)

        # Mean Squared Error for train and test sets
        train_errors[degree].append(np.mean(error_function(tmp_pred_train, y_train)))
        test_errors[degree].append(np.mean(error_function(tmp_pred_test, y_test)))


def calculate_estimator_bias_squared(pred_test):
    pred_test = np.array(pred_test)
    average_model_prediction = pred_test.mean(0)  # E[g(x)]

    # (E[g(x)] - f(x))^2, averaged across all trials
    return np.mean((average_model_prediction - fx(x_test)) ** 2)


def calculate_estimator_variance(pred_test):
    pred_test = np.array(pred_test)
    average_model_prediction = pred_test.mean(0)  # E[g(x)]

    # (g(x) - E[g(x)])^2, averaged across all trials
    return np.mean((pred_test - average_model_prediction) ** 2)


complexity_train_error = []
complexity_test_error = []
bias_squared = []
variance = []
for degree in model_poly_degrees:
    complexity_train_error.append(np.mean(train_errors[degree]))
    complexity_test_error.append(np.mean(test_errors[degree]))
    bias_squared.append(calculate_estimator_bias_squared(pred_test[degree]))
    variance.append(calculate_estimator_variance(pred_test[degree]))

best_model_degree = model_poly_degrees[np.argmin(complexity_test_error)]

plt.clf()

# Visualizations
ERROR_COLOR='grey'
## Plot Bias^2 + variance
plt.plot(model_poly_degrees, bias_squared, color='blue', label='$bias^2$')
plt.plot(model_poly_degrees, variance, color='green', label='variance')
plt.plot(model_poly_degrees, complexity_test_error, label='Testing Set Error', linewidth=3, color=ERROR_COLOR)
plt.axvline(best_model_degree, linestyle='--', color='black', label=f'Best Model(degree={best_model_degree})')

plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylim([0, 6]);  
plt.legend()
plt.title('Q1.4')
plt.savefig("Q1_4.png")

# Q1.5
# we have to make l2norm polyfit
def polyfit_with_l2(x, y, deg, alpha=0.0):
    # XTX+alphaI (beta) = XTY
    X = np.vander(x, deg+1) # (Sample, deg+1)
    left = X.T@X+alpha*np.eye(X.shape[1]) # (deg+1, deg+1)
    right = X.T@y # (deg+1, )
    return np.linalg.solve(left, right)
# Solve with l2norm
preds_test_with_10 = []
test_errors_with_10 = []

for dataset in range(n_datasets):

    # Simulate training/testing targets
    y_train = true_fn(x_train)
    y_test = true_fn(x_test)
    
    # degree => only 10
    # Loop over model complexities
    tmp_theta_hat = polyfit_with_l2(x_train, y_train, 10, alpha=1.0)

    # Test predictions
    tmp_pred_test = np.polyval(tmp_theta_hat, x_test)
    preds_test_with_10.append(tmp_pred_test)

    test_errors_with_10.append(np.mean(error_function(tmp_pred_test, y_test)))
    
complexity_test_error_with_10 = np.mean(test_errors_with_10)
bias_squared_with_10 = calculate_estimator_bias_squared(preds_test_with_10)
variance_with_10 = calculate_estimator_variance(preds_test_with_10)

print(f"BIAS SQUARED : Without Norm {bias_squared[10]}, With Norm {bias_squared_with_10}")
print(f"Variance  : Without Norm {variance[10]}, With Norm {variance_with_10}")
print(f"MSE  : Without Norm {complexity_test_error[10]}, With Norm {complexity_test_error_with_10}")