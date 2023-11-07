import os
import numpy as np
from skimage import io, color, feature
from sklearn.preprocessing import StandardScaler
import joblib
import psutil
import time  # Import the time module for timing and sleeping
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def perform_inference(image_paths, input_images, model):
    predictions = {}

    timestamps = []
    cpu_percentages = []
    memory_usages = []
    runtimes = []

    scaler = StandardScaler()
    input_images = scaler.fit_transform(input_images)

    # Performing inference on each input image
    for image_path, image in zip(image_paths, input_images):
        # Reshape the image as needed (e.g., if it's a single feature vector)
        image = image.reshape(1, -1)

        # Measureing system utilization before inference
        start_time = time.time()
        initial_cpu_usage = psutil.cpu_percent()
        initial_memory_usage = psutil.virtual_memory().percent

        # Make predictions using the loaded model
        prediction = model.predict(image)

        # Measure system utilization after inference
        end_time = time.time()
        final_cpu_usage = psutil.cpu_percent()
        final_memory_usage = psutil.virtual_memory().percent

        # Calculate runtime, CPU, and memory usage during inference
        runtime = end_time - start_time
        cpu_usage_during_inference = final_cpu_usage - initial_cpu_usage
        memory_usage_during_inference = final_memory_usage - initial_memory_usage
        
        print("Total Runtime of the program is:", runtime,"seconds")
        print("CPU Usage during the program is:", cpu_usage_during_inference,"KB")
        print("Memory Usage during the program is:", memory_usage_during_inference/1024,"KB")

        # Store the prediction in the dictionary
        predictions[image_path] = prediction[0]

        # Append metrics to lists
        timestamps.append(start_time)
        cpu_percentages.append(cpu_usage_during_inference)
        memory_usages.append(memory_usage_during_inference)
        runtimes.append(runtime)

    # Plot the graphs
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

    return predictions

if __name__ == "__main__":
    # Loading the trained logistic regression model from the ./Ml_models/ directory
    model_filename = "/home/scooby/Desktop/modelfile/logisticregression.pkl"
    loaded_model = joblib.load(model_filename)

    # Defining the path to the folder containing test images
    test_folder_path = '/home/scooby/Desktop/Crack Detection/Input_data/Test/'

    # Initializing empty lists to store image paths and data
    image_paths = []
    input_images = []
    for image_file in os.listdir(test_folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(test_folder_path, image_file)

            # Load and preprocess the image
            image = io.imread(image_path)
            gray_image = color.rgb2gray(image)
            features = feature.hog(gray_image)

            # Append the image path and data to the respective lists
            image_paths.append(image_path)
            input_images.append(features)

    # Perform inference with the loaded model 
    predictions = perform_inference(image_paths, input_images, loaded_model)

    # Print the predictions as a dictionary
    for image_path, prediction in predictions.items():
        print(f"Image File: {image_path}, Prediction: {prediction}")

    # Display the plots of system metrics
    plt.show()
