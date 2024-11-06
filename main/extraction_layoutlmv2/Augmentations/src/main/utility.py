
import cv2
import numpy as np
import random
import cv2
import numpy as np
import math

def draw_random_lines_across_image(path, num_lines):
    img = cv2.imread(path, 0)
    for _ in range(num_lines):
        # Generate random y-coordinate for the line
        y_coordinate = random.randint(0, img.shape[0])

        # Draw a horizontal line across the image at the generated y-coordinate
        img = cv2.line(img, (0, y_coordinate), (img.shape[1], y_coordinate), 0, 2)

    return img



def generate_random_circular_masks(image_shape, num_masks, min_distance, max_distance, min_radius, max_radius):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    center_distances = set()

    for _ in range(num_masks):
        # Generate random center coordinates within the second half of the image
        center_x = random.randint(image_shape[1] // 2, image_shape[1] - 1)
        center_y = random.randint(image_shape[0] // 2, image_shape[0] - 1)

        # Check if the new center is at least min_distance away from existing centers
        if all(np.sqrt((center_x - x)**2 + (center_y - y)**2) >= min_distance for x, y in center_distances):
            center_distances.add((center_x, center_y))

    for center_x, center_y in center_distances:
        # Generate random radius
        radius = random.randint(min_radius, max_radius)

        # Draw white circle on mask
        mask = cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)

    return mask

def masking(img_path):
    image = cv2.imread(img_path)

    # Define the number of circular masks
    num_masks = 2

    # Define the minimum and maximum distance between circular masks
    min_distance = 280
    max_distance = 300

    # Define the minimum and maximum radius for the circular masks
    min_radius = 100
    max_radius = 120

    # Generate random circular masks
    mask = generate_random_circular_masks(image.shape, num_masks, min_distance, max_distance, min_radius, max_radius)

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the circular masks
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Create a mask for the bounding box
    bbox_mask = np.zeros_like(image, dtype=np.uint8)
    bbox_mask[y:y+h, x:x+w] = 255

    # Set the region inside the bounding box to white
    result = cv2.bitwise_or(image, bbox_mask)
    return result


def angle_mask(img_path):
    image = cv2.imread(img_path)

    # Create a blank mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Specify the angle (in degrees) and the portion to be masked
    angle = 30
    mask_portion = 0.2  # Fraction of the image width to be masked

    # Calculate the end points of the line segment
    height, width = image.shape[:2]
    angle_rad = math.radians(angle)
    distance_from_corner = int(mask_portion * width)

    start_point = (0, 0)
    end_point = (distance_from_corner, height)

    # Draw a polygon on the mask
    pts = np.array([start_point, (0, height), end_point], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    # Toggle between masking and unmasking
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.bitwise_xor(image, masked_image)
    return result


def toggle_mask_all_corners(image_path, mask_size):
    # Read the image
    image = cv2.imread(image_path)

    # Create a blank mask
    mask = np.zeros_like(image, dtype=np.uint8)

    # Define the vertices of the triangles in all four corners
    height, width = image.shape[:2]

    # Bottom-left triangle
    vertices_bl = np.array([[0, height], [0, height - mask_size], [mask_size, height]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices_bl], (255, 255, 255))

    # Bottom-right triangle
    vertices_br = np.array([[width, height], [width, height - mask_size], [width - mask_size, height]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices_br], (255, 255, 255))

    # Top-left triangle
    vertices_tl = np.array([[0, 0], [0, mask_size], [mask_size, 0]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices_tl], (255, 255, 255))

    # Top-right triangle
    vertices_tr = np.array([[width, 0], [width, mask_size], [width - mask_size, 0]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices_tr], (255, 255, 255))

    # Toggle between masking and unmasking
    masked_image = cv2.bitwise_and(image, mask)
    unmasked_image = cv2.bitwise_xor(image, masked_image)

    return unmasked_image