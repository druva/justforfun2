import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import time

epoch_time = int(time.time())

# celsius_q = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], dtype=float)
# farenheit_a = np.array([32,33.8,35.6,37.4,39.2,41,42.8,44.6,46.4,48.2,50,51.8,53.6,55.4,57.2,59,60.8,62.6,64.4,66.2,68,69.8,71.6,73.4,75.2,77,78.8], dtype=float)
def generate_arrays(start, end):
    array1 = np.array(list(range(start, end + 1)))
    array2 = np.array([round((x * 1.8) + 32, 1) for x in array1])
    return array1, array2

# Example usage
celsius_q, farenheit_a = generate_arrays(0, 25)
for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, farenheit_a[i]))
with open(f"trainingset_{epoch_time}.txt", "a") as file:
    for i, c in enumerate(celsius_q):  # assuming c_values is a list of Celsius indices
        file.write(f"C{c} = F{farenheit_a[i]};\n")


#what is unit and what is input shape? and why we have to 1
#units out is output size? is it like shape?
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

#Why Sequential Model
model = tf.keras.Sequential([l0])

# What is compile? it is for preparing the model
# mean_squared_error - is inbuild loss function logic to calculate loss
# Why Adam? Why not others?
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))


#What is epochs? why 500 value
#what is fit?
for i in range(4):
  print(f"Running Fit ({i+1})n times")
  history = model.fit(celsius_q, farenheit_a, epochs=500, verbose=False)


test_celsius_q, test_farenheit_a = generate_arrays(25, 50)
result = model.predict(np.array(test_celsius_q))
# for i,c in enumerate(test_celsius_q):
#   print(f"C{c} = F{test_farenheit_a[i]}; Predicted= {np.round(result[i], decimals=1)} --  raw{result[i]}")

with open(f"results_{epoch_time}.txt", "a") as file:
    for i, c in enumerate(test_celsius_q):  # assuming c_values is a list of Celsius indices
        file.write(f"C{c} = F{test_farenheit_a[i]}; Predicted= {np.round(result[i], decimals=1)} --  raw{result[i]}\n")

#
model.save(f"model_{epoch_time}.h5")


plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()