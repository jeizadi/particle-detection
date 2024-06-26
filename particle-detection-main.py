# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:58:03 2024

@author: jeizadi
"""

import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from tkinter import filedialog

global image_path  # Path to the image
global image_name  # Reference to the file name of the selected video file
global original_image  # Reference to the OG image file
global image  # Reference to the opened image file
global image_roi  # Reference to the user-defined ROI for contour selection
global points  # Reference to the user-selected points
global data  # Reference to the dataframe


# Open a video file from a path and store the instance
def open_image_file():
    global image_path, image_name, image, original_image
    # Open a file dialog to select the video file
    image_file = filedialog.askopenfilename(title="Select an image", filetypes=[("JPEG files", "*.jpg"),
                                                                                ("PNG files", "*.png")])

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


# Click and drag to select the scale for the measurement
def select_point(event, x, y, flags, param):
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


# Color-threshold the image to isolate the red particulate in a binary
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
    lower = np.array([155, 25, 0])
    upper = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower, upper)
    
    result = cv2.bitwise_and(result, result, mask=mask_red)
    
    # Convert the result to a binary image
    binary_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary_result = cv2.threshold(binary_result, 1, 255, cv2.THRESH_BINARY)

    # Remove noise using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closing


# Fit contours to the binary using Contour Features in OpenCV and apply algorithms to fit shapes to each contour
# to determine enclosed area, perimeter, diameter, width, height, etc.
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
    areas = []  # Approximation contour identification
    perimeters = []  # Approximation contour identification
    widths = []  # Rotated rectangle width
    heights = []  # Rotated rectangle height
    diameters = []  # Diameter of bounded circle fit

    circle_overlay = image_roi.copy()
    rectangle_overlay = image_roi.copy()

    # Iterate through each contour
    for contour in contours:
        # Calculate the area and perimeter of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Draw a rectangle around the perimeter of the contour
        (x, y), (width, height), angle = cv2.minAreaRect(contour)
        box = cv2.boxPoints(((x, y), (width, height), angle))
        box = np.int0(box)
        cv2.drawContours(rectangle_overlay, [box], 0, (0, 0, 255), 2)

        # Minimum enclosing circle approximation
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = 2 * int(radius)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(circle_overlay, center, radius, (0, 255, 0), 2)

        if area > 0:
            particle_count += 1
            areas.append('%.3f' % (area * calibration_factor * calibration_factor))
            perimeters.append('%.3f' % (perimeter * calibration_factor))
            diameters.append('%.3f' % (diameter * calibration_factor))
            widths.append('%.3f' % (width * calibration_factor))
            heights.append('%.3f' % (height * calibration_factor))
        cv2.imshow("circles", circle_overlay)
        cv2.imshow("rectangles", rectangle_overlay)

    # data = pd.DataFrame(areas, perimeters, diameters, widths, heights)  # Add data to a dataframe for storage
    return (particle_count, areas, perimeters, diameters, widths, heights, contour_overlay, circle_overlay,
            rectangle_overlay)


'''
# Meant to enable the user to switch output units from in to mm to cm,etc. Disabled for this iteration of program
def process_data(Actual_unit, unit, areas):
    if Actual_unit != unit:
        if Actual_unit == 'mm': # Need to convert to inch
            areas = [float(area)/(25.4**2) for area in areas]
        else: # Need to convert to mm
            areas = [float(area)*(25.4**2) for area in areas]
    else:
        areas = [float(area) for area in areas]
    return areas
'''


# Can enable histogram creation with various bin sizes as a graphical output for the software
def create_histogram(data, x_scale, bin_size, file):
    bins = np.arange(0, x_scale + bin_size, bin_size)
    plt.hist(data, bins=bins, edgecolor='black')  # Plot the histogram
    plt.xlabel(f'Area (mm)')
    plt.ylabel('Frequency')
    plt.title('Red Particulate: {:.2f} to {:.2f}'.format(0, x_scale))
    plt.grid(True)
    plt.savefig(file)
    plt.show()


# Can enable create graphic to plot graphical summary and output this for the user
def create_graphic(areas, x_scale, file):
    mu, std = norm.fit(areas)
    fig, ax = plt.subplots()
    sns.histplot(data=areas, binwidth=x_scale/10, ax=ax, kde=True)
    ax.set_xlim(0, x_scale)
    plt.xlabel(f'Area (mm)')
    plt.title('Red Particulate: {:.2f} and {:.2f}'.format(mu, std))
    plt.savefig(file)  # Save the graphic
    plt.show()


# Creates a output .csv excel readable file with the various parameter outputted for each contour identified
# along with images of contours and various algorithms identification tracked
def output(particle_count, areas, perimeters, diameters, widths, heights, contour_overlay, circle_overlay,
           rectangle_overlay):
    folder_name = image_name.split('.')[0]
    filepath = os.path.join(image_path, folder_name)
    
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
        file.write('Area(mm), Perimeter(mm), Diameter(mm), Width(mm), Height(mm)\n')  # Write the header
        for area, perimeter, diameter, width, height in zip(areas, perimeters, diameters, widths, heights):
            file.write(str(area) + ',' + str(perimeter) + ',' + str(diameter) + ',' + str(width) + ',' + str(height) +
                       '\n')  # Write data for both columns

    '''
    # Save distribution graphics for specific size bins
    file_extension = filename + '_0_to_50mm.png'
    path = os.path.join(filepath, file_extension)
    create_graphic(areas, 50, path)
    
    file_extension = filename + '_0_to_5mm.png'
    path = os.path.join(filepath, file_extension)
    create_graphic(areas, 5, path)
    
    file_extension = filename + '_0_to_1mm.png'
    path = os.path.join(filepath, file_extension)
    create_graphic(areas, 1, path)
    '''

    # Save the image with the contours drawn 
    filename = filename + '_contours.png'
    os.chdir(filepath)
    cv2.imwrite(filename, contour_overlay)
    filename = filename + '_circles.png'
    os.chdir(filepath)
    cv2.imwrite(filename, circle_overlay)
    filename = filename + '_rectangles.png'
    os.chdir(filepath)
    cv2.imwrite(filename, rectangle_overlay)
    return


# Open an image file and set the scale by clicking and dragging a known distance. Select the region of interest. Will
# auto-threshold to identify contours and run min circle-fit and rotated rectangle-fit routines on binary image. Outputs
# include contour image, circles image, rectangles image and .csv file with area, perimeter, width, height, and diameter
def main(): 
    open_image_file()  # Open the image file
 
    global points
    points = []
    
    # bind select_point function to a window that will capture the mouse click
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_point)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    
    px = euclidean(points[0], points[1])
    cal_dim = float(input("Enter dimensions: "))
    # Actual_unit = input("Enter your unit(mm, inch): ")
    calibration_factor = cal_dim / px

    cv2.destroyAllWindows()
    
    # Preprocess the image
    binary_image = preprocess_image()
    
    # Count and measure the area of red particles
    particle_count, areas, perimeters, diameters, widths, heights, contour_overlay, circle_overlay, rectangle_overlay\
        = count_and_measure_area(binary_image, calibration_factor)

    '''
    unit = input("Enter your output unit(mm, inch): ")
    areas = process_data(Actual_unit, unit, areas) # Process the data for applicable units
    output(particle_count, areas, contour_overlay, unit)  # Output max area, total area, particle count. Save to file.  
    '''

    output(particle_count, areas, perimeters, diameters, widths, heights, contour_overlay, circle_overlay,
           rectangle_overlay)
        
    # De-allocate any associated memory usage   
    if cv2.waitKey(0) & 0xff == 27:  
        cv2.destroyAllWindows()  


if __name__ == "__main__":
    main()
    