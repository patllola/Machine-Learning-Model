import os
import cv2
import numpy as np
import joblib
import psutil
import time
import matplotlib.pyplot as plt

# Load the trained SVM model
model_filename = "/home/scooby/Desktop/best_svm_model_split_data.joblib"
svm_classifier = joblib.load(model_filename)

# Define the path to the folder containing test images
test_folder_path = '/home/scooby/Desktop/Crack Detection/Input_data/Test'
target_size = (64, 64)  # You can adjust the target size to match your preprocessing

# Function to load and preprocess test images
def load_and_preprocess_test_images(folder_path, target_size):
    test_images = []
    test_image_names = []

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        test_image = cv2.imread(image_path)
        test_image = cv2.resize(test_image, target_size)
        test_images.append(test_image)
        test_image_names.append(image_file)

    test_images = np.array(test_images)
    return test_images, test_image_names

# Load and preprocess test images
test_images, test_image_names = load_and_preprocess_test_images(test_folder_path, target_size)

# Flatten and normalize test images
test_images_flat = test_images.reshape(test_images.shape[0], -1) / 255.0

# Initialize empty lists to store the metrics
timestamps = []
cpu_percentages = []
memory_usages = []
runtimes = []

# Run your model within this block
start_time = time.time()
print("Initial Time: ", start_time)
initial_cpu_usage_rfc = psutil.cpu_percent()
print("Initial CPU Usage: ", initial_cpu_usage_rfc)
initial_memory_usage_rfc = psutil.virtual_memory().used
print("Initial Memory Usage: ", initial_memory_usage_rfc)

# Perform predictions using the trained SVM model
test_predictions = svm_classifier.predict(test_images_flat)

# Calculate runtime, CPU, and memory usage during inference
end_time = time.time()
final_cpu_usage_rfc = psutil.cpu_percent()
final_memory_usage_rfc = psutil.virtual_memory().used

runtime = end_time - start_time
cpu_usage_during_inference = final_cpu_usage_rfc - initial_cpu_usage_rfc
memory_usage_during_inference = final_memory_usage_rfc - initial_memory_usage_rfc

print("Total Runtime of the program is:", runtime)
print("CPU Usage during the program is:", cpu_usage_during_inference)
print("Memory Usage during the program is:", memory_usage_during_inference)

# Collect system metrics periodically
duration = 60  # Collect metrics for 60 seconds
interval = 5   # Collect metrics every 5 seconds

for i in range(duration // interval):
    timestamps.append(time.time())
    cpu_percentages.append(psutil.cpu_percent())
    memory_usages.append(psutil.virtual_memory().percent)
    runtimes.append(runtime)

    time.sleep(interval)

# Plot the metrics as graphs
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

# Create a dictionary to store the results (image name -> predicted label)
results = {}
for image_name, prediction in zip(test_image_names, test_predictions):
    results[image_name] = prediction

# Display the results
for image_name, prediction in results.items():
    print(f"Image: {image_name}, Predicted Label: {prediction}")
