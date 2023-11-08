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
# timestamps = []
# cpu_percentages = []
# memory_usages = []
# runtimes = []

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values to [0, 1]
    return img

def predict_single_image(image_path):
    img = preprocess_image(image_path)
    memory_info = psutil.virtual_memory()
    init_mem_use = psutil.virtual_memory().used
    start_time = time.time()
    psutil.cpu_percent(interval=None)
    prediction = model.predict(img)
    final_cpu_percent = psutil.cpu_percent(interval=None)
    final_time = time.time()
    final_mem_use = psutil.virtual_memory().used
    if prediction[0][0] > 0.5:
        return "Positive", final_time-start_time, final_mem_use-init_mem_use, final_cpu_percent
    else:
        return "Negative", final_time-start_time, final_mem_use-init_mem_use, final_cpu_percent


def predict_images_in_folder(folder_path):
    predictions = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            predicted_class, duration, mem_usage, cpu_usage_percent = predict_single_image(image_path)
            predictions[filename] = predicted_class, duration, mem_usage, cpu_usage_percent
    return predictions

def main(input_path):
    if os.path.isfile(input_path):  # Check if the input is a file
        predicted_class, duration, mem_usage, cpu_usage_percent = predict_single_image(input_path)
        print(f"Crack prediction: {predicted_class}\nTime: {duration} seconds\nMemory usage: {mem_usage}\nCPU usage: {cpu_usage_percent}%")

    elif os.path.isdir(input_path): # Check if the input is a directory
        folder_predictions = predict_images_in_folder(input_path)
        for filename, predicted_class in folder_predictions.items():
            # duration = 0
            # mem_usage = 0
            # cpu_usage_percent = 0
            print(f"Image {filename}: {predicted_class[0]}\nTime: {predicted_class[1]} seconds\nMemory usage: {predicted_class[2]}\nCPU usage: {predicted_class[3]}%")
            print(f"Image {filename}: {predicted_class}")
            
    else:
        print("Invalid input path.")

input_path = '/home/scooby/Desktop/Crack Detection/Input_data/Test'  # Replace with the path to your image or folder
main(input_path)

# Plot the metrics as graphs
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.plot(timestamps, cpu_percentages)
# plt.scatter(timestamps, cpu_percentages, marker='o', color='r')  # Add markers
# plt.title('CPU Utilization')
# plt.xlabel('Time')
# plt.ylabel('Percentage')

# plt.subplot(2, 2, 2)
# plt.plot(timestamps, memory_usages)
# plt.scatter(timestamps, memory_usages, marker='o', color='r')  # Add markers
# plt.title('Memory Usage')
# plt.xlabel('Time')
# plt.ylabel('Percentage')

# plt.subplot(2, 2, 3)
# plt.plot(timestamps, runtimes)
# plt.scatter(timestamps, runtimes, marker='o', color='r')  # Add markers
# plt.title('Runtime')
# plt.xlabel('Time')
# plt.ylabel('Seconds')

# plt.tight_layout()
# plt.show()
