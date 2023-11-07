import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import psutil
import time
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("/home/scooby/Desktop/modelfile/cnn2D_image_classification_model.h5")

# Initialize lists to store metrics
timestamps = []
cpu_percentages = []
memory_usages = []
runtimes = []

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values to [0, 1]
    return img

def predict_single_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        return "Positive"
    else:
        return "Negative"

def predict_images_in_folder(folder_path):
    predictions = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict_single_image(image_path)
            predictions[filename] = predicted_class
    return predictions

def measure_resource_usage():
    start_time = time.time()
    print("Initial Time: ", start_time)
    initial_cpu_usage = psutil.cpu_percent()
    print("Initial CPU Usage: ", initial_cpu_usage,"KB")
    initial_memory_usage = psutil.virtual_memory().used
    print("Initial Memory Usage: ", initial_memory_usage/1024,"KB")
    
    return start_time, initial_cpu_usage, initial_memory_usage

def main(input_path):
    if os.path.isfile(input_path):  # Check if the input is a file
        start_time, initial_cpu_usage, initial_memory_usage = measure_resource_usage()
        
        predicted_class = predict_single_image(input_path)
        print(f"The predicted class for the image is: {predicted_class}")
        
        end_time = time.time()
        print("End Time is:", end_time)
        final_cpu_usage = psutil.cpu_percent()
        print("Final CPU Usage is:", final_cpu_usage,"KB")
        final_memory_usage = psutil.virtual_memory().used
        print("Final Memory Usage is:", final_memory_usage/1024,"KB")
        
        # Calculate runtime, CPU, and memory usage during inference
        runtime = end_time - start_time
        cpu_usage_during_inference = final_cpu_usage - initial_cpu_usage
        memory_usage_during_inference = final_memory_usage - initial_memory_usage
        
        print("Total Runtime of the program is:", runtime,"seconds")
        print("CPU Usage during the program is:", cpu_usage_during_inference,"KB")
        print("Memory Usage during the program is:", memory_usage_during_inference,"KB")
        
#         timestamps.append(runtime)
        cpu_percentages.append(cpu_usage_during_inference)
        memory_usages.append(memory_usage_during_inference)
        runtimes.append(runtime)
        
    elif os.path.isdir(input_path): # Check if the input is a directory
        start_time, initial_cpu_usage, initial_memory_usage = measure_resource_usage()
        folder_predictions = predict_images_in_folder(input_path)
        for filename, predicted_class in folder_predictions.items():
            print(f"Image {filename}: {predicted_class}")
            
            end_time = time.time()
            print("End Time is:", end_time)
            final_cpu_usage = psutil.cpu_percent()
            print("Final CPU Usage is:", final_cpu_usage)
            final_memory_usage = psutil.virtual_memory().used
            print("Final Memory Usage is:", final_memory_usage)

            # Calculate runtime, CPU, and memory usage during inference
            runtime = end_time - start_time
            cpu_usage_during_inference = final_cpu_usage - initial_cpu_usage
            memory_usage_during_inference = final_memory_usage - initial_memory_usage

            print("Total Runtime of the program is:", runtime,"seconds")
            print("CPU Usage during the program is:", cpu_usage_during_inference,"KB")
            print("Memory Usage during the program is:", memory_usage_during_inference,"KB")

            # Append metrics to lists
            timestamps.append(start_time)
            cpu_percentages.append(cpu_usage_during_inference)
            memory_usages.append(memory_usage_during_inference)
            runtimes.append(runtime)
    else:
        print("Invalid input path.")

input_path = '/home/scooby/Desktop/Crack Detection/Input_data/Test/'  # Replace with the path to your image or folder
main(input_path)

# Plot the graphs
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(timestamps, cpu_percentages)
plt.scatter(timestamps, cpu_percentages, marker='o', color='r')  # Add markers
plt.title('CPU Utilization')
plt.xlabel('Time')
plt.ylabel('Percentage')

plt.subplot(2, 2, 2)
plt.plot(timestamps, memory_usages)
plt.scatter(timestamps, memory_usages, marker='o', color='r')  # Add markers
plt.title('Memory Usage')
plt.xlabel('Time')
plt.ylabel('Percentage')

plt.subplot(2, 2, 3)
plt.plot(timestamps, runtimes)
plt.scatter(timestamps, runtimes, marker='o', color='r')  # Add markers
plt.title('Runtime')
plt.xlabel('Time')
plt.ylabel('Seconds')

plt.tight_layout()
plt.show()
