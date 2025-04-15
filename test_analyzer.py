import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, color

# Import functions from other modules
from nuclei_detection import detect_nuclei
from myotube_detection import detect_myotubes
from nuclei_myotube_relationship import analyze_nuclei_myotube_relationship
from visualization_reporting import create_enhanced_visualization, generate_html_report, generate_csv_report

def test_analysis_pipeline(image_path, output_dir=None):
    """
    Test the complete analysis pipeline on a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_dir : str, optional
        Directory to save output images and results
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "validation_results")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Testing analysis pipeline on image: {image_path}")
    print(f"Results will be saved to: {output_dir}")
    
    # Step 1: Detect nuclei
    print("\nStep 1: Detecting nuclei...")
    nuclei_results = detect_nuclei(image_path, output_dir, visualize=True)
    print(f"Detected {nuclei_results['nuclei_count']} nuclei")
    
    # Step 2: Detect myotubes
    print("\nStep 2: Detecting myotubes...")
    myotube_results = detect_myotubes(image_path, output_dir, visualize=True)
    print(f"Detected {myotube_results['myotube_count']} myotube regions")
    print(f"Myotube area: {myotube_results['myotube_area_percentage']:.2f}% of image")
    
    # Step 3: Analyze relationship
    print("\nStep 3: Analyzing nuclei-myotube relationship...")
    relationship_results = analyze_nuclei_myotube_relationship(image_path, output_dir, visualize=True)
    print(f"Total nuclei: {relationship_results['total_nuclei']}")
    print(f"Nuclei within myotubes: {relationship_results['nuclei_within_myotubes']} ({relationship_results['percentage_within_myotubes']:.2f}%)")
    print(f"Nuclei outside myotubes: {relationship_results['nuclei_outside_myotubes']} ({100-relationship_results['percentage_within_myotubes']:.2f}%)")
    
    # Step 4: Create enhanced visualization
    print("\nStep 4: Creating enhanced visualization...")
    enhanced_vis_path = create_enhanced_visualization(image_path, relationship_results, output_dir)
    
    # Step 5: Generate reports
    print("\nStep 5: Generating reports...")
    html_report_path = generate_html_report(image_path, relationship_results, output_dir)
    csv_report_path = generate_csv_report(relationship_results, output_dir)
    
    # Save validation results
    validation_results = {
        "nuclei_count": nuclei_results['nuclei_count'],
        "myotube_count": myotube_results['myotube_count'],
        "myotube_area_percentage": myotube_results['myotube_area_percentage'],
        "total_nuclei": relationship_results['total_nuclei'],
        "nuclei_within_myotubes": relationship_results['nuclei_within_myotubes'],
        "nuclei_outside_myotubes": relationship_results['nuclei_outside_myotubes'],
        "percentage_within_myotubes": relationship_results['percentage_within_myotubes']
    }
    
    # Write validation results to text file
    validation_text = "Myotube Analysis Validation Results\n"
    validation_text += "================================\n\n"
    validation_text += f"Image: {os.path.basename(image_path)}\n\n"
    validation_text += "Nuclei Detection:\n"
    validation_text += f"- Nuclei count: {validation_results['nuclei_count']}\n\n"
    validation_text += "Myotube Detection:\n"
    validation_text += f"- Myotube count: {validation_results['myotube_count']}\n"
    validation_text += f"- Myotube area: {validation_results['myotube_area_percentage']:.2f}% of image\n\n"
    validation_text += "Nuclei-Myotube Relationship:\n"
    validation_text += f"- Total nuclei: {validation_results['total_nuclei']}\n"
    validation_text += f"- Nuclei within myotubes: {validation_results['nuclei_within_myotubes']} ({validation_results['percentage_within_myotubes']:.2f}%)\n"
    validation_text += f"- Nuclei outside myotubes: {validation_results['nuclei_outside_myotubes']} ({100-validation_results['percentage_within_myotubes']:.2f}%)\n\n"
    validation_text += "Output Files:\n"
    validation_text += f"- Enhanced visualization: {os.path.basename(enhanced_vis_path)}\n"
    validation_text += f"- HTML report: {os.path.basename(html_report_path)}\n"
    validation_text += f"- CSV report: {os.path.basename(csv_report_path)}\n"
    
    validation_file_path = os.path.join(output_dir, "validation_results.txt")
    with open(validation_file_path, "w") as f:
        f.write(validation_text)
    
    print("\nValidation complete!")
    print(f"Validation results saved to: {validation_file_path}")
    
    return validation_results

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
        output_dir = "validation_results"
    
    # Run the test
    validation_results = test_analysis_pipeline(image_path, output_dir)
