import cv2 as cv
import matplotlib.pyplot as plt

# Load the image - change .jpg to .png if your image is a png file
img = cv.imread('crop_field.jpg')

# Check if image loaded correctly
if img is None:
    print("ERROR: Image not found!")
    print("Make sure crop_field.jpg is in the same folder as this file")
else:
    print("✅ Image loaded successfully!")
    print("Image size (height x width):", img.shape[:2])

    # Convert color so it looks correct
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Show the image
    plt.figure(figsize=(12, 7))
    plt.imshow(img_rgb)
    plt.title('Hover mouse over the crop area - note the coordinates at bottom')
    plt.show()