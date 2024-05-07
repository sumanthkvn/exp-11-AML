import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Disable eager execution
tf.compat.v1.disable_eager_execution()

ops.reset_default_graph()

# Create graph
sess = tf.compat.v1.Session()

# Load the data
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Split data into training and validation sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare batch size
batch_size = 50

# Initialize placeholders
x_data = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A_ridge = tf.Variable(tf.random.normal(shape=[1,1]))
b_ridge = tf.Variable(tf.random.normal(shape=[1,1]))

A_lasso = tf.Variable(tf.random.normal(shape=[1,1]))
b_lasso = tf.Variable(tf.random.normal(shape=[1,1]))

# Declare model operations
model_output_ridge = tf.add(tf.matmul(x_data, A_ridge), b_ridge)
model_output_lasso = tf.add(tf.matmul(x_data, A_lasso), b_lasso)

# Declare the Ridge loss function
ridge_param = tf.constant(1.)
ridge_loss = tf.reduce_mean(tf.square(A_ridge))
ridge_loss += tf.reduce_mean(tf.square(y_target - model_output_ridge))
ridge_loss = tf.expand_dims(tf.add(ridge_loss, tf.multiply(ridge_param, ridge_loss)), 0)

# Declare the Lasso loss function
lasso_param = tf.constant(1.)
lasso_loss = tf.reduce_mean(tf.abs(A_lasso))
lasso_loss += tf.reduce_mean(tf.square(y_target - model_output_lasso))
lasso_loss = tf.expand_dims(tf.add(lasso_loss, tf.multiply(lasso_param, lasso_loss)), 0)

# Declare optimizers
my_opt_ridge = tf.compat.v1.train.GradientDescentOptimizer(0.001)
train_step_ridge = my_opt_ridge.minimize(ridge_loss)

my_opt_lasso = tf.compat.v1.train.GradientDescentOptimizer(0.001)
train_step_lasso = my_opt_lasso.minimize(lasso_loss)

# Initialize variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# Training loop for Ridge regression
loss_vec_ridge = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step_ridge, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(ridge_loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec_ridge.append(temp_loss[0])
    if (i+1)%300==0:
        print('Ridge Regression - Step #' + str(i+1) + ' A = ' + str(sess.run(A_ridge)) + ' b = ' + str(sess.run(b_ridge)))
        print('Loss = ' + str(temp_loss))

# Training loop for Lasso regression
loss_vec_lasso = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step_lasso, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(lasso_loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec_lasso.append(temp_loss[0])
    if (i+1)%300==0:
        print('Lasso Regression - Step #' + str(i+1) + ' A = ' + str(sess.run(A_lasso)) + ' b = ' + str(sess.run(b_lasso)))
        print('Loss = ' + str(temp_loss))

# Compute predictions on the test set for Ridge regression
test_predictions_ridge = sess.run(model_output_ridge, feed_dict={x_data: np.transpose([x_vals_test])})

# Compute accuracy metrics for Ridge regression
mse_ridge = mean_squared_error(y_vals_test, test_predictions_ridge)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_vals_test, test_predictions_ridge)
r2_ridge = r2_score(y_vals_test, test_predictions_ridge)

print("Ridge Regression - Mean Squared Error (MSE):", mse_ridge)
print("Ridge Regression - Root Mean Squared Error (RMSE):", rmse_ridge)
print("Ridge Regression - Mean Absolute Error (MAE):", mae_ridge)
print("Ridge Regression - R^2 Score:", r2_ridge)

# Compute predictions on the test set for Lasso regression
test_predictions_lasso = sess.run(model_output_lasso, feed_dict={x_data: np.transpose([x_vals_test])})

# Compute accuracy metrics for Lasso regression
mse_lasso = mean_squared_error(y_vals_test, test_predictions_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_vals_test, test_predictions_lasso)
r2_lasso = r2_score(y_vals_test, test_predictions_lasso)

print("Lasso Regression - Mean Squared Error (MSE):", mse_lasso)
print("Lasso Regression - Root Mean Squared Error (RMSE):", rmse_lasso)
print("Lasso Regression - Mean Absolute Error (MAE):", mae_lasso)
print("Lasso Regression - R^2 Score:", r2_lasso)

# Close the session
sess.close()