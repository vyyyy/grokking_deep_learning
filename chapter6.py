# Chapter 6 | Building your first deep neural network

# Creating a matrix or two in Python
import numpy as np

# input data
streetlights = np.array([
    [ 1, 0, 1 ],
    [ 0, 1, 1 ],
    [ 0, 0, 1 ],
    [ 1, 1, 1 ],
    [ 0, 1, 1 ],
    [ 1, 0, 1 ]
])

# output data
walk_vs_stop = np.array([ 
    [ 0 ],
    [ 1 ],
    [ 0 ],
    [ 1 ],
    [ 1 ],
    [ 0 ] 
])


# Building a neural network
import numpy as np

weights = np.array([0.5,0.48,-0.7])
alpha = 0.1

# input data
streetlights = np.array([
    [ 1, 0, 1 ],
    [ 0, 1, 1 ],
    [ 0, 0, 1 ],
    [ 1, 1, 1 ],
    [ 0, 1, 1 ],
    [ 1, 0, 1 ]
])

# output data
walk_vs_stop = np.array( [ 0, 1, 0, 1, 1, 0 ] )

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for iteration in range(20):
    prediction = input.dot(weights)
    error = (goal_prediction - prediction) ** 2
    delta = prediction - goal_prediction
    weights = weights - (alpha * (input * delta))
    print("Error:" + str(error) + " Prediction:" + str(prediction))


# Learning the whole dataset
# The neural network has been learning only one streetlight. Don’t we want it to learn them all?
import numpy as np

weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1

# input data
streetlights = np.array([
    [ 1, 0, 1 ],
    [ 0, 1, 1 ],
    [ 0, 0, 1 ],
    [ 1, 1, 1 ],
    [ 0, 1, 1 ],
    [ 1, 0, 1 ]
])

# output data
walk_vs_stop = np.array( [ 0, 1, 0, 1, 1, 0 ] )

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]

        prediction = input.dot(weights)

        error = (goal_prediction - prediction) ** 2
        error_for_all_lights += error

        delta = prediction - goal_prediction
        weights = weights - (alpha * (input * delta))
        print("Prediction:" + str(prediction))
    print("Error:" + str(error_for_all_lights) + "\n")


# Your first deep neural network
import numpy as np

np.random.seed(1)

# sets all negative numbers to 0
def relu(x):
    return (x > 0) * x

alpha = 0.2
hidden_size = 4

# input data
streetlights = np.array([
 [ 1, 0, 1 ],
 [ 0, 1, 1 ],
 [ 0, 0, 1 ],
 [ 1, 1, 1 ]])

# output data
walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T

weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

layer_0 = streetlights[0]
layer_1 = relu(np.dot(layer_0,weights_0_1))
layer_2 = np.dot(layer_1,weights_1_2)


# Backpropagation in code
import numpy as np

np.random.seed(1)

# Returns x if x > 0; returns 0 otherwise
def relu(x):
    return (x > 0) * x

# Returns 1 for input > 0; returns 0 otherwise
def relu2deriv(output):
    return output>0

alpha = 0.2
hidden_size = 4

weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)
        
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

        layer_2_delta = (walk_vs_stop[i:i+1] - layer_2)
        layer_1_delta=layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if(iteration % 10 == 9):
        print("Error:" + str(layer_2_error))


# One interation of backpropagation
# 1. Initializing the network's weights and data
import numpy as np

np.random.seed(1)

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output > 0

streetlights = np.array([
 [ 1, 0, 1 ],
 [ 0, 1, 1 ],
 [ 0, 0, 1 ],
 [ 1, 1, 1 ]])

walk_stop = np.array([[ 1, 1, 0, 0]]).T

alpha = 0.2
hidden_size = 3

weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

# 2. Predict + Compare: Making a prediction, and calculating the output error and delta
layer_0 = streetlights[0:1]
layer_1 = np.dot(layer_0,weights_0_1)
layer_1 = relu(layer_1)
layer_2 = np.dot(layer_1,weights_1_2)

error = (layer_2-walk_stop[0:1])**2

layer_2_delta=(layer_2-walk_stop[0:1])

# 3. Learn: Backpropagating from layer_2 to layer_1
layer_0 = streetlights[0:1]
layer_1 = np.dot(layer_0,weights_0_1)
layer_1 = relu(layer_1)
layer_2 = np.dot(layer_1,weights_1_2)

error = (layer_2-walk_stop[0:1])**2

layer_2_delta=(layer_2-walk_stop[0:1])

layer_1_delta=layer_2_delta.dot(weights_1_2.T)
layer_1_delta *= relu2deriv(layer_1)

# 4. Learn: Generating weight_deltas, and updating weights
layer_0 = streetlights[0:1]
layer_1 = np.dot(layer_0,weights_0_1)
layer_1 = relu(layer_1)
layer_2 = np.dot(layer_1,weights_1_2)

error = (layer_2-walk_stop[0:1])**2

layer_2_delta=(layer_2-walk_stop[0:1])

layer_1_delta=layer_2_delta.dot(weights_1_2.T)
layer_1_delta *= relu2deriv(layer_1)

weight_delta_1_2 = layer_1.T.dot(layer_2_delta)
weight_delta_0_1 = layer_0.T.dot(layer_1_delta)

weights_1_2 -= alpha * weight_delta_1_2
weights_0_1 -= alpha * weight_delta_0_1


# Putting it all together
import numpy as np

np.random.seed(1)

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output>0

streetlights = np.array([
 [ 1, 0, 1 ],
 [ 0, 1, 1 ],
 [ 0, 0, 1 ],
 [ 1, 1, 1 ]])

walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T

alpha = 0.2
hidden_size = 4

weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)

        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

        layer_2_delta = (layer_2 - walk_vs_stop[i:i+1])
        layer_1_delta=layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

    if(iteration % 10 == 9):
        print("Error:" + str(layer_2_error))

#Error:0.634231159844
#Error:0.358384076763
#Error:0.0830183113303
#Error:0.0064670549571
#Error:0.000329266900075
#Error:1.50556226651e-05
