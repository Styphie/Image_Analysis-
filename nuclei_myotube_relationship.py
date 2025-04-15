import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, color

# Import functions from other modules
from nuclei_detection import detect_nuclei
from myotube_detection import detect_myotubes

def analyze_nuclei_myotube_relationship(image_path, output_dir=None, visualize=True):
    """
    Analyze the relationship between nuclei and myotubes in a fluorescence microscopy image.
    
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
        Dictionary containing relationship metrics
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
    
    # Detect nuclei
    print("Detecting nuclei...")
    nuclei_results = detect_nuclei(image_path, output_dir, visualize=False)
    
    # Detect myotubes
    print("Detecting myotubes...")
    myotube_results = detect_myotubes(image_path, output_dir, visualize=False)
    
    # Get nuclei labels and myotube mask
    nuclei_labels = nuclei_results['labels']
    myotube_mask = myotube_results['myotube_mask']
    labeled_myotubes = myotube_results['labeled_myotubes']
    
    # Count nuclei within myotubes
    nuclei_within_myotubes = 0
    nuclei_outside_myotubes = 0
    
    # Get unique nuclei labels (excluding background)
    unique_nuclei = np.unique(nuclei_labels)
    unique_nuclei = unique_nuclei[unique_nuclei > 0]  # Exclude background (0)
    
    # For each nucleus, check if it's within a myotube
    nuclei_centroids = []
    nuclei_in_myotube = []
    
    for nucleus_label in unique_nuclei:
        # Get the region properties for this nucleus
        nucleus_mask = nuclei_labels == nucleus_label
        nucleus_props = measure.regionprops(nucleus_mask.astype(int))
        
        if nucleus_props:
            # Get centroid of the nucleus
            centroid = nucleus_props[0].centroid
            nuclei_centroids.append(centroid)
            
            # Check if the centroid is within a myotube
            y, x = int(centroid[0]), int(centroid[1])
            if y < myotube_mask.shape[0] and x < myotube_mask.shape[1] and myotube_mask[y, x] > 0:
                nuclei_within_myotubes += 1
                nuclei_in_myotube.append(True)
            else:
                nuclei_outside_myotubes += 1
                nuclei_in_myotube.append(False)
    
    # Calculate percentage of nuclei within myotubes
    total_nuclei = nuclei_within_myotubes + nuclei_outside_myotubes
    percentage_within_myotubes = (nuclei_within_myotubes / total_nuclei) * 100 if total_nuclei > 0 else 0
    
    if visualize:
        # Create visualization
        # Create a color-coded image showing nuclei and myotubes
        # Blue for nuclei outside myotubes, green for nuclei within myotubes, red for myotubes
        visualization = np.zeros((*image_rgb.shape[:2], 3), dtype=np.uint8)
        
        # Add myotubes in red
        visualization[myotube_mask > 0, 0] = 255  # Red channel
        
        # Add nuclei
        for i, centroid in enumerate(nuclei_centroids):
            y, x = int(centroid[0]), int(centroid[1])
            if y < visualization.shape[0] and x < visualization.shape[1]:
                # Draw a circle at the nucleus centroid
                if nuclei_in_myotube[i]:
                    # Green for nuclei within myotubes
                    cv2.circle(visualization, (x, y), 5, (0, 255, 0), -1)
                else:
                    # Blue for nuclei outside myotubes
                    cv2.circle(visualization, (x, y), 5, (0, 0, 255), -1)
        
        # Create figure for visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Nuclei detection
        nuclei_vis = color.label2rgb(nuclei_labels, image=image_rgb, bg_label=0)
        axes[0, 1].imshow(nuclei_vis)
        axes[0, 1].set_title(f'Nuclei Detection (Count: {total_nuclei})')
        axes[0, 1].axis('off')
        
        # Myotube detection
        myotube_vis = color.label2rgb(labeled_myotubes, image=image_rgb, bg_label=0, alpha=0.5)
        axes[1, 0].imshow(myotube_vis)
        axes[1, 0].set_title(f'Myotube Detection (Count: {myotube_results["myotube_count"]})')
        axes[1, 0].axis('off')
        
        # Relationship visualization
        axes[1, 1].imshow(visualization)
        axes[1, 1].set_title(f'Nuclei-Myotube Relationship\n'
                           f'Within: {nuclei_within_myotubes} ({percentage_within_myotubes:.2f}%)\n'
                           f'Outside: {nuclei_outside_myotubes} ({100-percentage_within_myotubes:.2f}%)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, 'nuclei_myotube_relationship.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Total nuclei: {total_nuclei}")
        print(f"Nuclei within myotubes: {nuclei_within_myotubes} ({percentage_within_myotubes:.2f}%)")
        print(f"Nuclei outside myotubes: {nuclei_outside_myotubes} ({100-percentage_within_myotubes:.2f}%)")
        print(f"Saved visualization to {output_path}")
    
    # Prepare results
    results = {
        'total_nuclei': int(total_nuclei),
        'nuclei_within_myotubes': int(nuclei_within_myotubes),
        'nuclei_outside_myotubes': int(nuclei_outside_myotubes),
        'percentage_within_myotubes': float(percentage_within_myotubes),
        'myotube_count': int(myotube_results['myotube_count']),
        'myotube_area_percentage': float(myotube_results['myotube_area_percentage']),
        'nuclei_centroids': nuclei_centroids,
        'nuclei_in_myotube': nuclei_in_myotube
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
    
    # Analyze relationship
    results = analyze_nuclei_myotube_relationship(image_path, output_dir)
    
    print(f"Total nuclei: {results['total_nuclei']}")
    print(f"Nuclei within myotubes: {results['nuclei_within_myotubes']} ({results['percentage_within_myotubes']:.2f}%)")
    print(f"Nuclei outside myotubes: {results['nuclei_outside_myotubes']} ({100-results['percentage_within_myotubes']:.2f}%)")
