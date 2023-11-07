import os
import cv2
import numpy as np
import joblib
from skimage import io, color, feature
import psutil
import time
import matplotlib.pyplot as plt
# Load the trained K-NN model
model_filename = "/home/scooby/Desktop/modelfile/knn_model.joblib"
knn_classifier = joblib.load(model_filename)
timestamps = []
cpu_percentages = []
memory_usages = []
runtimes = []

start_time = time.time()
print("Initial Time: ", start_time)
initial_cpu_usage = psutil.cpu_percent()
print("Initial CPU Usage: ", initial_cpu_usage,"KB")
initial_memory_usage = psutil.virtual_memory().used
print("Initial Memory Usage: ", initial_memory_usage/1024,"KB")

# Defing predict_single_image function with HOG feature extraction
def predict_single_image(image_path, model):
    img = io.imread(image_path)
        
    # Ensure all pixel values are non-negative (shift to [0, 255] range)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    img_gray = color.rgb2gray(img)
    hog_features = feature.hog(img_gray, block_norm='L2-Hys', pixels_per_cell=(16, 16))
     # Reshape the HOG features to match the shape expected by the model
    hog_features = hog_features.reshape(1, -1)
     # Make a prediction using the model
    prediction = model.predict(hog_features)
    end_time = time.time()
    final_cpu_usage = psutil.cpu_percent()
    final_memory_usage = psutil.virtual_memory().used

    runtime = end_time - start_time
    cpu_usage_during_inference = final_cpu_usage - initial_cpu_usage
    memory_usage_during_inference = final_memory_usage - initial_memory_usage

    print("Total Runtime of the program is:", runtime,"seconds")
    print("CPU Usage during the program is:", cpu_usage_during_inference,"KB")
    print("Memory Usage during the program is:", memory_usage_during_inference,"KB")

    # Collect system metrics periodically
    duration = 60  # Collect metrics for 60 seconds
    interval = 5   # Collect metrics every 5 seconds

    for i in range(duration // interval):
        timestamps.append(time.time())
        cpu_percentages.append(psutil.cpu_percent())
        memory_usages.append(psutil.virtual_memory().percent)
        runtimes.append(runtime)

        time.sleep(interval)

    # Plot the  graphs
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(timestamps, cpu_percentages)
    plt.title('CPU Utilization')
    plt.xlabel('Time')
    plt.ylabel('Percentage')

    plt.subplot(2, 2, 2)
    plt.plot(timestamps, memory_usages)
    plt.title('Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Percentage')

    plt.subplot(2, 2, 3)
    plt.plot(timestamps, runtimes)
    plt.title('Runtime')
    plt.xlabel('Time')
    plt.ylabel('Seconds')

    plt.tight_layout()
    # Display the plots
    plt.show()
    return prediction[0]



def predict_images_in_directory(directory_path, model):
    predictions = {}
    
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            prediction = predict_single_image(image_path, model)
            predictions[image_file] = prediction
    
    return predictions




# You can pass either a single image path or a directory containing images
input_path = "/home/scooby/Desktop/Crack Detection/Input_data/Test/"
if os.path.isfile(input_path):
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        prediction = predict_single_image(input_path, knn_classifier)
        print(f"Image: {input_path}, Predicted Label: {prediction}")
    else:
        print("Invalid file format. Please provide a valid image file.")
elif os.path.isdir(input_path):
    predictions = predict_images_in_directory(input_path, knn_classifier)
    for image_name, prediction in predictions.items():
        print(f"Image: {image_name}, Predicted Label: {prediction}")
else:
    print("Invalid input. Please provide a valid image file or directory.")    
