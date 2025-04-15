import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation, color, feature

def detect_nuclei(image_path, output_dir=None, visualize=True):
    """
    Detect and count nuclei in a fluorescence microscopy image.
    
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
        Dictionary containing nuclei count and other metrics
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
    
    # Extract blue channel (nuclei)
    blue_channel = image_rgb[:, :, 2]  # In RGB, blue is channel 2
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(blue_channel, (5, 5), 0)
    
    # Apply Otsu's thresholding to segment nuclei
    threshold_value, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu's threshold value: {threshold_value}")
    
    # Apply morphological operations to clean up the binary image
    # First, remove small noise with erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    
    # Then dilate to restore nuclei size
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    # Apply distance transform for watershed segmentation
    dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    
    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Find local maxima (markers) for watershed
    coordinates = feature.peak_local_max(dist_transform, min_distance=7, labels=dilated)
    
    # Create a boolean mask of the same shape as the image
    local_max = np.zeros_like(dist_transform, dtype=bool)
    
    # Mark the coordinates of local maxima as True
    for coord in coordinates:
        local_max[coord[0], coord[1]] = True
    
    # Label the markers
    markers = measure.label(local_max)
    
    # Apply watershed segmentation
    labels = segmentation.watershed(-dist_transform, markers, mask=dilated)
    
    # Count nuclei (number of regions)
    nuclei_count = np.max(labels)
    
    if visualize:
        # Create visualization
        # Create a color-coded segmentation image
        segmented_image = color.label2rgb(labels, image=image_rgb, bg_label=0)
        
        # Create figure for visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Blue channel
        axes[0, 1].imshow(blue_channel, cmap='gray')
        axes[0, 1].set_title('Blue Channel (Nuclei)')
        axes[0, 1].axis('off')
        
        # Binary image after thresholding
        axes[1, 0].imshow(dilated, cmap='gray')
        axes[1, 0].set_title('Binary Image after Morphology')
        axes[1, 0].axis('off')
        
        # Segmented nuclei
        axes[1, 1].imshow(segmented_image)
        axes[1, 1].set_title(f'Segmented Nuclei (Count: {nuclei_count})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, 'nuclei_detection_results.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Detected {nuclei_count} nuclei")
        print(f"Saved visualization to {output_path}")
    
    # Prepare results
    results = {
        'nuclei_count': int(nuclei_count),
        'threshold_value': float(threshold_value),
        'labels': labels
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
    
    # Detect nuclei
    results = detect_nuclei(image_path, output_dir)
    
    print(f"Total nuclei count: {results['nuclei_count']}")
