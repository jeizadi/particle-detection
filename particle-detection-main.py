# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:58:03 2024

@author: jeizadi
"""
import math

import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from tkinter import filedialog

global image_path # Path to the image
global image_name # Reference to the file name of the selected video file
global original_image # Reference to the OG image file
global image # Reference to the opened image file
global image_roi # Reference to the user-defined ROI for contour selection

# Open a video file from a path and store the instance
def open_image_file():
    global image_path, image_name, image, original_image
    # Open a file dialog to select the video file
    image_file = filedialog.askopenfilename(title="Select an image", filetypes=[("JPEG files", "*.jpg"),("PNG files", "*.png")])

    if not image_file:
        return  # No file selected

    image_path, image_name = os.path.split(image_file)
    
    # Load the image
    image = cv2.imread(image_file)
    original_image = cv2.imread(image_file)
    
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]
    if original_width > 800 or original_height > 600:
        resize_image(800, 600)
        
# Resize an image to fit within w = 800 & h = 600 
def resize_image(max_width, max_height):
    global image
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / float(original_height)

    # Calculate the new dimensions to fit within the specified limits
    new_width = min(original_width, max_width)
    new_height = int(new_width / aspect_ratio)

    if new_height > max_height:
        # If the height exceeds the limit, resize based on height
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    image = cv2.resize(image, (new_width, new_height))
    
def select_point(event,x,y,flags,param):
    global image, points
    
    # Record starting (x,y) coordinates on left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        points.append([ix, iy])
            
    # Record ending (x,y) coordinates on left mouse bottom release
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x,y))
        
        cv2.line(image, points[0], points[1], (36,255,12), 2)
        cv2.imshow("image", image)

def preprocess_image():
    global image, image_roi
    
    # Allow the user to define a Region of Interest (ROI)
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")

    # Crop the image based on the ROI
    x, y, w, h = map(int, roi)
    image_roi = image[y:y+h, x:x+w]

    # Convert the image to HSV color space
    result = image_roi.copy()
    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    
    # Threshold for red color
    lower = np.array([155,25,0])
    upper = np.array([179,255,255])
    mask_red = cv2.inRange(hsv, lower, upper)
    
    result = cv2.bitwise_and(result, result, mask=mask_red)
    
    # Convert the result to a binary image
    binary_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary_result = cv2.threshold(binary_result, 1, 255, cv2.THRESH_BINARY)

    # Remove noise using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closing

def count_and_measure_area(binary_image, calibration_factor):
    global image_roi
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    contour_overlay = image_roi.copy()
    cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)  # -1 means draw all contours
    cv2.imshow("contours", contour_overlay)
    # Initialize variables to count particles and total area
    particle_count = 0
    areas = []
    
    # Iterate through each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        area = area*calibration_factor
        if area > 0:
            particle_count += 1
            areas.append('%.3f'%(area))
                
    return particle_count, areas, contour_overlay

def process_data(Actual_unit, unit, areas):
    if Actual_unit != unit:
        if Actual_unit == 'mm': # Need to convert to inch
            areas = [float(area)/(25.4**2) for area in areas]
        else: # Need to convert to mm
            areas = [float(area)*(25.4**2) for area in areas]
    else:
        areas = [float(area) for area in areas]
    return areas
        
def create_histogram(data, x_scale, bin_size, units, file):
    bins = np.arange(0, x_scale + bin_size, bin_size)
    plt.hist(data, bins=bins, edgecolor='black') # Plot the histogram
    plt.xlabel(f'Area ({units}\u00b2)')
    plt.ylabel('Frequency')
    plt.title('Coating Particulate: {:.2f} to {:.2f}'.format(0, x_scale))
    plt.grid(True)
    plt.savefig(file) # Save the histogram
    plt.show() # Display the plot

def create_graphic(areas, x_scale, unit, file):
    mu, std = norm.fit(areas)
    fig, ax = plt.subplots()
    sns.histplot(data=areas, binwidth=x_scale/10, ax=ax, kde=True)
    ax.set_xlim(0, x_scale)
    plt.xlabel(f'Area ({unit}\u00b2)')
    plt.title('Coating Particulate: {:.2f} and {:.2f}'.format(mu, std))
    plt.savefig(file) # Save the graphic
    plt.show()
    
def output(particle_count, areas, contour_overlay, unit): 
    # Print the total area and max area
    total_area = sum(areas)
    max_area = max(areas)
    
    # Print the results
    print(f"Number of Red Particles: {particle_count}")
    print(f"Total Area: {total_area:.3f}")
    print(f"Max Area: {max_area:.3f}")
    
    foldername = image_name.split('.')[0] 
    filepath = os.path.join(image_path, foldername)
    
    # Create the file path if one does not exist already
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Get the current date and time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        
    # Create filenames stamped with the current time and date
    filename = os.path.splitext(image_name)[0] # Get the file name without an extension
    filename = filename + "_" + formatted_datetime
    
    # Save the raw data to a file in the output folder path
    file_extension = filename + '.csv'
    path = os.path.join(filepath, file_extension)
    with open(path, 'w', newline='') as file:
        file.write('Areas Given In ' + unit.upper() + ',Area-Derived Diameter\n')  # Write the header
        for area in areas:
            diameter = 2 * math.sqrt(area / math.pi)
            file.write(str(area) + ',' + str(diameter) + '\n')  # Write data for both columns
    
    # Save distribution graphics for specific size bins
    file_extension = filename + '_0_to_50' + unit + '.png'
    path = os.path.join(filepath, file_extension)
    create_graphic(areas, 50, unit, path)
    
    file_extension = filename + '_0_to_5' + unit + '.png'
    path = os.path.join(filepath, file_extension)
    create_graphic(areas, 5, unit, path)
    
    file_extension = filename + '_0_to_1' + unit + '.png'
    path = os.path.join(filepath, file_extension)
    create_graphic(areas, 1, unit, path)
    
    # Save the image with the contours drawn 
    file_extension = filename + '_contours.png'
    
    return
    
def main(): 
    open_image_file() # Open the image file
 
    global points
    points = []
    
    # bind select_point function to a window that will capture the mouse click
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_point)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    
    px= euclidean(points[0], points[1])
    Actual_dimensions= float(input("Enter dimensions: "))
    Actual_unit = input("Enter your unit(mm, inch): ")
    
    calibration_factor= Actual_dimensions/px
    calibration_factor= calibration_factor**2
    
    cv2.destroyAllWindows()
    
    # Preprocess the image
    binary_image = preprocess_image()
    
    # Count and measure the area of red particles
    particle_count, areas, contour_overlay = count_and_measure_area(binary_image, calibration_factor)
    
    unit = input("Enter your output unit(mm, inch): ")
    
    areas = process_data(Actual_unit, unit, areas) # Process the data for applicable units
    output(particle_count, areas, contour_overlay, unit) # Output max area, total area, particle count. Save to file.     
        
    # De-allocate any associated memory usage   
    if cv2.waitKey(0) & 0xff == 27:  
        cv2.destroyAllWindows()  

if __name__ == "__main__":
    main()