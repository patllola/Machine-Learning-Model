import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import psutil
import time
import os

timestamps = []
cpu_percentages = []
battery_percentages = []
memory_usages = []
runtimes = []

# Run your model within this block
start_time = time.time()
print("Initial Time: ", start_time)
initial_cpu_usage = psutil.cpu_percent()
print("Initial CPU Usage: ", initial_cpu_usage,"KB")
initial_memory_usage = psutil.virtual_memory().used
print("Initial Memory Usage: ", initial_memory_usage/1024,"KB")
# Load the saved model
loaded_model = load_model('/home/scooby/Desktop/modelfile/fine_tuned_model')

# Function to preprocess input image
def preprocess_image(image_path, target_size):
    images = []
    image_names = []

    for image_file in os.listdir(image_path):
        if image_file.lower().endswith(('.jpg')):
            image_file_path = os.path.join(image_path, image_file)
            image = cv2.imread(image_file_path)
            image = cv2.resize(image, target_size)
            images.append(image)
            image_names.append(image_file)

    images = np.array(images)
    return images, image_names

target_size = (150, 150)

# Function to perform inference
def perform_inference(image_path, target_size):
    # Preprocess the input image
    images, image_names = preprocess_image(image_path, target_size)
    
    # Perform inference using the loaded model
    predictions = loaded_model.predict(images)
    
    for image_name, prediction in zip(image_names, predictions):
        predicted_label = "Positive" if prediction[0] > 0.5 else "Negative"
        print(f"Image: {image_name}, Predicted Label: {predicted_label}, Predicted Value: {prediction[0]}")
        
    return predictions, predicted_label

# Example usage
image_path = '/home/scooby/Desktop/Crack Detection/Input_data/Test'
predicted_label, predictions = perform_inference(image_path, target_size)
print("Prediction value:", predictions)
print("Predicted Label:", predicted_label)

end_time = time.time()
print("End Time is:", end_time)
final_cpu_usage = psutil.cpu_percent()
print("Final CPU Usage is:", final_cpu_usage,"KB")
final_memory_usage = psutil.virtual_memory().used
print("Final Memory Usage is:", final_memory_usage/1024, "KB")

# Calculate runtime, CPU, and memory usage during inference
runtime = end_time - start_time
cpu_usage_during_inference = final_cpu_usage - initial_cpu_usage
memory_usage_during_inference = final_memory_usage - initial_memory_usage

print("Total Runtime of the program is:", runtime,"seconds")
print("CPU Usage during the program is:", cpu_usage_during_inference,"KB")
print("Memory Usage during the program is:", memory_usage_during_inference,"KB")

# Append metrics to lists
# timestamps.append(start_time)
cpu_percentages.append(cpu_usage_during_inference)
memory_usages.append(memory_usage_during_inference)
runtimes.append(runtime)

# # Plot the metrics as graphs
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
