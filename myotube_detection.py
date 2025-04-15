import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation, color

def detect_myotubes(image_path, output_dir=None, visualize=True):
    """
    Detect and segment myotubes in a fluorescence microscopy image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_dir : str, optional
        Directory to save output images and results
    visualize : bool, optional
        Whether to create and save visualization
        
    Returns:
    --------
    dict
        Dictionary containing myotube count, area, and other metrics
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract red channel (myotubes)
    red_channel = image_rgb[:, :, 0]  # In RGB, red is channel 0
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
    
    # Apply adaptive thresholding to segment myotubes
    # Myotubes have varying intensity, so adaptive thresholding works better than global
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        21,  # Block size
        -5   # Constant subtracted from mean
    )
    
    # Apply morphological operations to clean up the binary image
    # First, remove small noise with opening (erosion followed by dilation)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Then close gaps in myotubes with closing (dilation followed by erosion)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small objects (noise)
    labeled_image = measure.label(closed)
    regions = measure.regionprops(labeled_image)
    
    # Filter regions by area to remove small noise
    min_area = 100  # Minimum area threshold
    filtered_binary = np.zeros_like(closed)
    
    myotube_count = 0
    total_myotube_area = 0
    
    for region in regions:
        if region.area >= min_area:
            myotube_count += 1
            total_myotube_area += region.area
            for coord in region.coords:
                filtered_binary[coord[0], coord[1]] = 255
    
    # Calculate myotube metrics
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    myotube_area_percentage = (total_myotube_area / image_area) * 100
    
    if visualize:
        # Create visualization
        # Label each myotube region
        labeled_myotubes = measure.label(filtered_binary)
        segmented_image = color.label2rgb(labeled_myotubes, image=image_rgb, bg_label=0, alpha=0.5)
        
        # Create figure for visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Red channel
        axes[0, 1].imshow(red_channel, cmap='gray')
        axes[0, 1].set_title('Red Channel (Myotubes)')
        axes[0, 1].axis('off')
        
        # Binary image after processing
        axes[1, 0].imshow(filtered_binary, cmap='gray')
        axes[1, 0].set_title('Processed Binary Image')
        axes[1, 0].axis('off')
        
        # Segmented myotubes
        axes[1, 1].imshow(segmented_image)
        axes[1, 1].set_title(f'Segmented Myotubes (Count: {myotube_count})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, 'myotube_detection_results.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Detected {myotube_count} myotube regions")
        print(f"Total myotube area: {total_myotube_area} pixels")
        print(f"Myotube area percentage: {myotube_area_percentage:.2f}%")
        print(f"Saved visualization to {output_path}")
    
    # Prepare results
    results = {
        'myotube_count': int(myotube_count),
        'total_myotube_area': float(total_myotube_area),
        'myotube_area_percentage': float(myotube_area_percentage),
        'myotube_mask': filtered_binary,
        'labeled_myotubes': labeled_myotubes if 'labeled_myotubes' in locals() else measure.label(filtered_binary)
    }
    
    return results

if __name__ == "__main__":
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image path
        image_path = "B1 high (2).png"
    
    # Check if output directory is provided as command line argument
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        # Default output directory
        output_dir = "results"
    
    # Detect myotubes
    results = detect_myotubes(image_path, output_dir)
    
    print(f"Myotube count: {results['myotube_count']}")
    print(f"Myotube area percentage: {results['myotube_area_percentage']:.2f}%")
