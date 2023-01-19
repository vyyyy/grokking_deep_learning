## Chapter 4: Introduction to neural learning: gradient descent

# Let's measure the error with mean squared error
knob_weight = 0.5
input = 0.5
goal_pred = 0.8
pred = input * knob_weight
error = (pred - goal_pred) ** 2
print(error) # 0.30


# Learning using the hot and cold method
# 1. An empty network
weight = 0.1

# learning rate
lr = 0.01

def neural_network(input, weight):
    prediction = input * weight
    return prediction

# 2. Predict: Making a prediction and evaluating error
number_of_toes = [8.5]
win_or_lose_binary = [1] # (won!!!)
input = number_of_toes[0]
true = win_or_lose_binary[0]

pred = neural_network(input, weight)
error = (pred - true) ** 2
print(error) # 0.023

# 3. Compare: Making a prediction with a higher weight and evaluating error
weight = 0.1
lr = 0.01
weight_plus_lr = 0.11
p_up = neural_network(input, weight+lr) # prediction with higher weight 8.5 * (0.11) -> 0.935
e_up = (p_up - true) ** 2 # error with higher weight (0.935 - 1.0) ** 2 -> 0.004225
print(e_up) # 0.004

# 4. Compare: Making a prediction with a lower weight and evaluating error
weight = 0.1
lr = 0.01
weight_minus_lr = 0.09
p_dn = neural_network(input, weight-lr) # 8.5 * 0.09 -> 0.765
e_dn = (p_dn - true) ** 2 # (0.765 - 1.0) ** 2 -> 0.055
print(e_dn) # 0.055

# 5. Compare + Learn: Comparing the errors and setting the new weight
prev_error = 0.023
prev_weight = 0.1
# weight + learning rate
new_error = 0.004
new_weight = 0.11

# learning rate adjusted weight reduces error that is less than previous weight's error
if(error > e_dn or error > e_up):
    # decreased weight error is less than increased weight error
    if(e_dn < e_up):
        weight -= lr
    # decreased weight error is greater than increased weight error
    if(e_dn > e_up):
        weight += lr


# Hot and cold learning in Jupyter Notebook
weight = 0.5
input = 0.5
goal_prediction = 0.8
step_amount = 0.001

for iteration in range(1101):
    # predict
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2
    print("Error:" + str(error) + " Prediction:" + str(prediction))

    # compare
    up_prediction = input * (weight + step_amount)
    up_error = (goal_prediction - up_prediction) ** 2
    down_prediction = input * (weight - step_amount)
    down_error = (goal_prediction - down_prediction) ** 2

    # learn
    if(down_error < up_error):
        weight = weight - step_amount

    if(down_error > up_error):
        weight = weight + step_amount
# Error:0.3025 Prediction:0.25
# Error:0.30195025 Prediction:0.2505
# ....
# Error:2.50000000033e-07 Prediction:0.7995
# Error:1.07995057925e-27 Prediction:0.8


# Calculating both direction and amount from error using gradient descent
weight = 0.5
goal_pred = 0.8
input = 0.5

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    # pure error -> (prediction - goal_prediction)
    # scaling, negative reversal, stopping -> * input
    direction_and_amount = (pred - goal_pred) * input
    weight = weight - direction_and_amount
    print("Error:" + str(error) + " Prediction:" + str(pred))
# Error:0.3025 Prediction:0.25
# Error:0.17015625 Prediction:0.3875
# Error:0.095712890625 Prediction:0.490625
# ...
# Error:1.7092608064e-05 Prediction:0.79586567925
# Error:9.61459203602e-06 Prediction:0.796899259437
# Error:5.40820802026e-06 Prediction:0.797674444578


# One iteration of gradient descent
# 1. An empty network
weight = 0.1
alpha = 0.01
def neural_network(input, weight):
    prediction = input * weight
    return prediction

# 2. Predict: Making a prediction and evaluating error
number_of_toes = [8.5]
win_or_lose_binary = [1] # (won!!!)
input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]
pred = neural_network(input, weight)
error = (pred - goal_pred) ** 2

# 3. Compare: Calculating the node delta and putting it on the output node
number_of_toes = [8.5]
win_or_lose_binary = [1] # (won!!!)
input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]
pred = neural_network(input, weight)
error = (pred - goal_pred) ** 2

delta = pred - goal_pred

# 4. Learn: Calculating the weight delta and putting it on the weight
number_of_toes = [8.5]
win_or_lose_binary = [1] # (won!!!)
input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]
pred = neural_network(input, weight)
error = (pred - goal_pred) ** 2
delta = pred - goal_pred

weight_delta = input * delta

# 5. Learn: Updating the weight
number_of_toes = [8.5]
win_or_lose_binary = [1] # (won!!!)
input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]
pred = neural_network(input, weight)
error = (pred - goal_pred) ** 2
delta = pred - goal_pred
weight_delta = input * delta

alpha = 0.01
weight -= weight_delta * alpha


# Let's watch several steps of learning
weight, goal_pred, input = (0.0, 0.8, 1.1)
for iteration in range(4):
    print("-----\nWeight:" + str(weight))
    pred = input * weight
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    weight_delta = delta * input
    weight = weight - weight_delta
    print("Error:" + str(error) + " Prediction:" + str(pred))
    print("Delta:" + str(delta) + " Weight Delta:" + str(weight_delta))
# -----
# Weight:0.0
# Error:0.64 Prediction:0.0
# Delta:-0.8 Weight Delta:-0.88
# -----
# Weight:0.88
# Error:0.028224 Prediction:0.968
# Delta:0.168 Weight Delta:0.1848
# -----
# Weight:0.6952
# Error:0.0012446784 Prediction:0.76472
# Delta:-0.03528 Weight Delta:-0.038808
# -----
# Weight:0.734008
# Error:5.489031744e-05 Prediction:0.8074088
# Delta:0.0074088 Weight Delta:0.00814968


# What is weight_delta, really?
weight = 0.5
goal_pred = 0.8
input = 0.5
for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * input
    weight = weight - direction_and_amount
    print("Error:" + str(error) + " Prediction:" + str(pred))


# Learning is adjusting the weight to reduce the error to 0.

# Breaking gradient descent

# Divergence
# Updating weight = weight - (input * (pred - goal_pred))
# If the input is large, this can make the weight update large even when the error is
# small. What happens when you have a large weight update and a small error? The network
# overcorrects. If the new error is even bigger, the network overcorrects even more. This
# causes divergence. 


# Alpha
# Solves the problem of overcorrecting weight updates.
# Multiply the weight by a number between 0 and 1 called alpha.
# formula -> weight = weight - (alpha * derivative)

# Run this code with larger input using alpha
weight = 0.5
goal_pred = 0.8
input = 2
alpha = 0.1

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    derivative = input * (pred - goal_pred)
    weight = weight - (alpha * derivative)

    print("Error:" + str(error) + " Prediction:" + str(pred))

# Error:0.04 Prediction:1.0
# Error:0.0144 Prediction:0.92
# Error:0.005184 Prediction:0.872
# ...
# Error:1.14604719983e-09 Prediction:0.800033853319
# Error:4.12576991939e-10 Prediction:0.800020311991
# Error:1.48527717099e-10 Prediction:0.800012187195
