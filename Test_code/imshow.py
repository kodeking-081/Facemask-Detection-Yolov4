def imShow(path):
    import cv2
    import matplotlib.pyplot as plt
    
    # Read the image using OpenCV
    image = cv2.imread(path)
    
    # Check if image is loaded properly
    if image is None:
        print(f"Error: Unable to load image at {path}")
        return
    
    # Resize the image (scaling by a factor of 3)
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)
    
    # Create a figure with desired size
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    
    # Hide axis and display image
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.show()

#this function is used to load and display image using OpenCV and matplotlinb.