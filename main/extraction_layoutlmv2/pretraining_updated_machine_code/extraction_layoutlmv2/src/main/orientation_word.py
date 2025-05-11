import numpy as np

def calculate_orientation(word_bbox):
    # Calculate centroids of characters
    centroids = [(bbox[0] + bbox[2]) / 2 for bbox in word_bbox]

    # Calculate angle between horizontal axis and line connecting first and last centroids
    delta_y = centroids[-1] - centroids[0]
    delta_x = word_bbox[-1][2] - word_bbox[0][0]
    angle = np.arctan2(delta_y, delta_x)

    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees

# Example usage
# word_bbox = [[110, 446, 124, 455], [127, 446, 131, 455], [152, 446, 187, 455], [243, 446, 260, 455], [264, 446, 276, 455]]  # Example bounding box coordinates of characters [(xmin, ymin, xmax, ymax)]
# word_bbox = [[110, 446, 124, 455], [33, 438, 48, 479], [32, 399, 47, 435]]
word_bbox = [[197, 270, 251, 277], [254, 270, 284, 277], [286, 270, 300, 277]]#, [243, 446, 260, 455], [264, 446, 276, 455]]  # Example bounding box coordinates of characters [(xmin, ymin, xmax, ymax)]
word_bbox = [[453, 139, 631, 159], [302, 168, 354, 182]]#, [243, 446, 260, 455], [264, 446, 276, 455]]  # Example bounding box coordinates of characters [(xmin, ymin, xmax, ymax)]
orientation = calculate_orientation(word_bbox)
print("Orientation of the word:", orientation)
