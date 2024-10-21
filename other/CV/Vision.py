from collections import defaultdict
import numpy as np
from auto_mark import *
def calculate_yuv_range(color_data, buffer):
    # Dictionary to hold aggregated color data per color category
    color_groups = defaultdict(list)
    
    # Group colors by their categories
    for color_dict in color_data:
        for color_name, values in color_dict.items():
            for value in values:
                origin_point = value['origin_point']
                radius = value['radius']
                color = value['color']
                
                # Add color information to the respective color group
                color_groups[color_name].append((origin_point, radius, color))
    
    # Calculate the YUV range for each color group
    yuv_ranges = {}
    for color_name, items in color_groups.items():
        # Extract the YUV values
        yuv_values = np.array([item[2] for item in items])
        
        # Calculate the average for each YUV component
        avg_yuv = np.mean(yuv_values, axis=0)
        y,u,v = avg_yuv
        # Define the range by applying the buffer
        
        min_yuv = [y-buffer,u-buffer,v-buffer]
        max_yuv = [y+buffer,u+buffer,v+buffer]
        min_yuv = np.clip(min_yuv, 0, 255)
        max_yuv = np.clip(max_yuv, 0, 255)

        print(min_yuv,max_yuv)
        # Store the YUV range as (min, max)
        yuv_ranges[color_name] = (min_yuv, max_yuv)
    
    return yuv_ranges



def identify_colors_in_image(yuv_ranges, image):
    """
    Identify colors in an image based on predefined YUV ranges and overlay the mask on the original image.

    Parameters:
    - yuv_ranges: A dictionary containing the YUV ranges for each color category.
    - image: The input image in BGR format.

    Returns:
    - A new image where each pixel is labeled with the identified color overlay.
    """
    # Convert the image from BGR to YUV
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Create an output image to display identified colors
    output_image = np.zeros_like(image)

    # Iterate through each color range and apply a mask to find matching pixels
    for color_name, (min_yuv, max_yuv) in yuv_ranges.items():
        # Create a mask for pixels within the current YUV range
        mask = cv2.inRange(yuv_image, min_yuv, max_yuv)

        # For visualization: Assign a color to each identified category (BGR format)
        if color_name == 'yellow':
            color_bgr = (0, 255, 255)  # Yellow in BGR
        elif color_name == 'pink':
            color_bgr = (255, 0, 255)  # Pink in BGR
        elif color_name == 'green':
            color_bgr = (0, 255, 0)    # Green in BGR
        elif color_name == 'blue':
            color_bgr = (255, 0, 0)    # Blue in BGR
        else:
            color_bgr = (255, 255, 255)  # Default to white for unknown colors

        # Create a colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color_bgr

        # Overlay the colored mask on the output image
        output_image = cv2.addWeighted(output_image, 1.0, colored_mask, 1.0, 0)

    # Convert the output image to RGB for display
    output_rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return output_rgb_image

if __name__ == "__main__":
    randomrules = RandomRules()
    image = cv2.imread("00000.jpg")
    height, width, _ = image.shape
    rectangles = utils.read_rectangles_from_file("00000.txt", height, width)
    mark_data = randomrules.get_mark_data(image,rectangles)
    hsv_ranges = calculate_yuv_range(mark_data,50)
    # print(hsv_ranges)


    # # Example usage

    # # Load an example image
    # image = cv2.imread('00000.jpg')

    # # Identify colors in the image
    # identified_image = identify_colors_in_image(hsv_ranges, image)

    # # Display the original and identified images
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Identified Colors', identified_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
