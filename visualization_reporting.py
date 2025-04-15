import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, color
import csv
import datetime
import json

# Import functions from other modules
from nuclei_detection import detect_nuclei
from myotube_detection import detect_myotubes
from nuclei_myotube_relationship import analyze_nuclei_myotube_relationship

def create_enhanced_visualization(image_path, results, output_dir=None):
    """
    Create enhanced visualization of the analysis results.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    results : dict
        Dictionary containing analysis results
    output_dir : str, optional
        Directory to save output images and results
        
    Returns:
    --------
    str
        Path to the saved visualization
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Nuclei visualization
    nuclei_vis = plt.imread(os.path.join(output_dir, 'nuclei_detection_results.png'))
    axes[0, 1].imshow(nuclei_vis)
    axes[0, 1].set_title(f'Nuclei Detection (Count: {results["total_nuclei"]})', fontsize=14)
    axes[0, 1].axis('off')
    
    # Myotube visualization
    myotube_vis = plt.imread(os.path.join(output_dir, 'myotube_detection_results.png'))
    axes[1, 0].imshow(myotube_vis)
    axes[1, 0].set_title(f'Myotube Detection (Count: {results["myotube_count"]})', fontsize=14)
    axes[1, 0].axis('off')
    
    # Relationship visualization
    relationship_vis = plt.imread(os.path.join(output_dir, 'nuclei_myotube_relationship.png'))
    axes[1, 1].imshow(relationship_vis)
    axes[1, 1].set_title('Nuclei-Myotube Relationship', fontsize=14)
    axes[1, 1].axis('off')
    
    # Add summary text
    plt.figtext(0.5, 0.01, 
                f'Summary: {results["total_nuclei"]} nuclei detected, {results["percentage_within_myotubes"]:.2f}% within myotubes. '
                f'Myotube area: {results["myotube_area_percentage"]:.2f}% of image.',
                ha='center', fontsize=14, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save visualization
    output_path = os.path.join(output_dir, 'enhanced_visualization.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved enhanced visualization to {output_path}")
    
    return output_path

def generate_html_report(image_path, results, output_dir=None):
    """
    Generate an HTML report of the analysis results.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    results : dict
        Dictionary containing analysis results
    output_dir : str, optional
        Directory to save output report
        
    Returns:
    --------
    str
        Path to the saved HTML report
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get current date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Myotube Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .results-summary {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            .results-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }}
            .results-table th, .results-table td {{
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            .results-table th {{
                background-color: #f2f2f2;
            }}
            .results-table tr:nth-child(even) {{
                background-color: #f8f8f8;
            }}
            .visualization {{
                margin-bottom: 30px;
                text-align: center;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
                font-size: 0.9em;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Myotube Analysis Report</h1>
                <p>Analysis performed on: {date_str}</p>
                <p>Image: {os.path.basename(image_path)}</p>
            </div>
            
            <div class="results-summary">
                <h2>Analysis Summary</h2>
                <p>This report presents the results of automated analysis of myotubes and nuclei in the provided microscopy image.</p>
                <ul>
                    <li><strong>Total nuclei detected:</strong> {results["total_nuclei"]}</li>
                    <li><strong>Nuclei within myotubes:</strong> {results["nuclei_within_myotubes"]} ({results["percentage_within_myotubes"]:.2f}%)</li>
                    <li><strong>Nuclei outside myotubes:</strong> {results["nuclei_outside_myotubes"]} ({100-results["percentage_within_myotubes"]:.2f}%)</li>
                    <li><strong>Myotube count:</strong> {results["myotube_count"]}</li>
                    <li><strong>Myotube area:</strong> {results["myotube_area_percentage"]:.2f}% of image area</li>
                </ul>
            </div>
            
            <h2>Detailed Results</h2>
            <table class="results-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Total nuclei</td>
                    <td>{results["total_nuclei"]}</td>
                    <td>Total number of nuclei detected in the image</td>
                </tr>
                <tr>
                    <td>Nuclei within myotubes</td>
                    <td>{results["nuclei_within_myotubes"]}</td>
                    <td>Number of nuclei located within myotube regions</td>
                </tr>
                <tr>
                    <td>Nuclei outside myotubes</td>
                    <td>{results["nuclei_outside_myotubes"]}</td>
                    <td>Number of nuclei located outside myotube regions</td>
                </tr>
                <tr>
                    <td>Percentage within myotubes</td>
                    <td>{results["percentage_within_myotubes"]:.2f}%</td>
                    <td>Percentage of nuclei located within myotube regions</td>
                </tr>
                <tr>
                    <td>Myotube count</td>
                    <td>{results["myotube_count"]}</td>
                    <td>Number of distinct myotube regions detected</td>
                </tr>
                <tr>
                    <td>Myotube area percentage</td>
                    <td>{results["myotube_area_percentage"]:.2f}%</td>
                    <td>Percentage of image area covered by myotubes</td>
                </tr>
            </table>
            
            <h2>Visualizations</h2>
            
            <div class="visualization">
                <h3>Enhanced Visualization</h3>
                <img src="enhanced_visualization.png" alt="Enhanced Visualization">
                <p>Comprehensive visualization showing original image, nuclei detection, myotube detection, and their relationship.</p>
            </div>
            
            <div class="visualization">
                <h3>Nuclei Detection</h3>
                <img src="nuclei_detection_results.png" alt="Nuclei Detection">
                <p>Visualization of the nuclei detection process and results.</p>
            </div>
            
            <div class="visualization">
                <h3>Myotube Detection</h3>
                <img src="myotube_detection_results.png" alt="Myotube Detection">
                <p>Visualization of the myotube detection process and results.</p>
            </div>
            
            <div class="visualization">
                <h3>Nuclei-Myotube Relationship</h3>
                <img src="nuclei_myotube_relationship.png" alt="Nuclei-Myotube Relationship">
                <p>Visualization showing the spatial relationship between nuclei and myotubes.</p>
            </div>
            
            <div class="footer">
                <p>This report was generated automatically by the Myotube Analyzer tool.</p>
                <p>© 2025 Myotube Analyzer</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    output_path = os.path.join(output_dir, 'myotube_analysis_report.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved HTML report to {output_path}")
    
    return output_path

def generate_csv_report(results, output_dir):
    """
    Generate a CSV report of the analysis results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    output_dir : str
        Directory to save output report
        
    Returns:
    --------
    str
        Path to the saved CSV report
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for CSV
    data = [
        ['Metric', 'Value', 'Description'],
        ['Total nuclei', results["total_nuclei"], 'Total number of nuclei detected in the image'],
        ['Nuclei within myotubes', results["nuclei_within_myotubes"], 'Number of nuclei located within myotube regions'],
        ['Nuclei outside myotubes', results["nuclei_outside_myotubes"], 'Number of nuclei located outside myotube regions'],
        ['Percentage within myotubes', f"{results['percentage_within_myotubes']:.2f}%", 'Percentage of nuclei located within myotube regions'],
        ['Myotube count', results["myotube_count"], 'Number of distinct myotube regions detected'],
        ['Myotube area percentage', f"{results['myotube_area_percentage']:.2f}%", 'Percentage of image area covered by myotubes']
    ]
    
    # Save CSV report
    output_path = os.path.join(output_dir, 'myotube_analysis_results.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    print(f"Saved CSV report to {output_path}")
    
    return output_path

def generate_json_report(results, output_dir):
    """
    Generate a JSON report of the analysis results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    output_dir : str
        Directory to save output report
        
    Returns:
    --------
    str
        Path to the saved JSON report
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for JSON
    # Remove non-serializable data (like numpy arrays)
    json_results = {
        'total_nuclei': int(results["total_nuclei"]),
        'nuclei_within_myotubes': int(results["nuclei_within_myotubes"]),
        'nuclei_outside_myotubes': int(results["nuclei_outside_myotubes"]),
        'percentage_within_myotubes': float(results["percentage_within_myotubes"]),
        'myotube_count': int(results["myotube_count"]),
        'myotube_area_percentage': float(results["myotube_area_percentage"]),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Save JSON report
    output_path = os.path.join(output_dir, 'myotube_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Saved JSON report to {output_path}")
    
    return output_path

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
    
    # Run the complete analysis pipeline
    print(f"Analyzing image: {image_path}")
    
    # Detect nuclei
    nuclei_results = detect_nuclei(image_path, output_dir)
    
    # Detect myotubes
    myotube_results = detect_myotubes(image_path, output_dir)
    
    # Analyze relationship
    relationship_results = analyze_nuclei_myotube_relationship(image_path, output_dir)
    
    # Create enhanced visualization
    enhanced_vis_path = create_enhanced_visualization(image_path, relationship_results, output_dir)
    
    # Generate reports
    html_report_path = generate_html_report(image_path, relationship_results, output_dir)
    csv_report_path = generate_csv_report(relationship_results, output_dir)
    json_report_path = generate_json_report(relationship_results, output_dir)
    
    print("\nAnalysis Summary:")
    print(f"Total nuclei: {relationship_results['total_nuclei']}")
    print(f"Nuclei within myotubes: {relationship_results['nuclei_within_myotubes']} ({relationship_results['percentage_within_myotubes']:.2f}%)")
    print(f"Nuclei outside myotubes: {relationship_results['nuclei_outside_myotubes']} ({100-relationship_results['percentage_within_myotubes']:.2f}%)")
    print(f"Myotube count: {relationship_results['myotube_count']}")
    print(f"Myotube area: {relationship_results['myotube_area_percentage']:.2f}% of image")
    
    print("\nOutput files:")
    print(f"Enhanced visualization: {enhanced_vis_path}")
    print(f"HTML report: {html_report_path}")
    print(f"CSV report: {csv_report_path}")
    print(f"JSON report: {json_report_path}")
