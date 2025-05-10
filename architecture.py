import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

def create_climate_prediction_architecture(save_path='climate_prediction_architecture.png', dpi=300):
    """
    Creates a detailed architecture diagram for the Climate Action Prediction System.
    This diagram illustrates the flow of data, user input, processing, and predictions
    for flood, drought, and landslide risks using AI/ML models and rule-based systems.
    
    Parameters:
    save_path (str): Path to save the diagram image
    dpi (int): Resolution of the saved image
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='white')  # Increased figure size for clarity
    
    # Color scheme - using climate-related colors
    colors = {
        'data': '#E3F2FD',  # Light blue for data
        'input': '#E0F7FA',  # Light cyan for user input
        'processing': '#E8F5E9',  # Light green for processing
        'flood': '#BBDEFB',  # Blue for flood prediction
        'drought': '#FFF3E0',  # Orange for drought prediction
        'landslide': '#E8EAF6',  # Purple for landslide prediction 
        'output': '#F3E5F5',  # Light purple for output/results
        'api': '#F9FBE7',  # Light yellow for API services
        'arrows': '#455A64',  # Dark blue-gray for connections
        'text': '#263238',  # Darker blue-gray for text
        'border': '#90A4AE',  # Gray for borders
        'ml_detail': '#F5F5F5',  # Very light gray for ML details
        'tensorflow': '#FF9E80',  # TensorFlow color
        'sklearn': '#B2DFDB',  # Scikit-learn color
        'rule_based': '#E1F5FE'  # Rule-based systems color
    }
    
    # Box properties
    box_width = 2.2  # Slightly wider boxes
    box_height = 0.8  # Slightly taller boxes
    corner_radius = 0.1
    border_width = 2
    
    # Define component positions with more spacing to prevent overlap
    components = {
        # Data Layer
        'historical_data': {'x': 1, 'y': 9.5, 'width': 3.2, 'height': 1.2, 'color': colors['data'], 
                      'label': 'Historical Climate Data\n- Flood Past Data (1980-2015)\n- State & Terrain Mapping\n- Rainfall Records by Region'},
        
        'generated_data': {'x': 6, 'y': 9.5, 'width': 3.2, 'height': 1.2, 'color': colors['data'], 
                     'label': 'Generated Climate Data\n- Flood Generation Data\n- State-wise Monthly Rainfall\n- Terrain Classification Data'},
        
        # User Input Layer
        'user_input': {'x': 1, 'y': 7.5, 'width': box_width, 'height': box_height, 'color': colors['input'], 
                    'label': 'User Input Form\n- Location\n- Date'},
        'location_input': {'x': 4, 'y': 7.5, 'width': box_width, 'height': box_height, 'color': colors['input'], 
                       'label': 'Location Parsing\n- City/State Mapping'},
        'date_input': {'x': 7, 'y': 7.5, 'width': box_width, 'height': box_height, 'color': colors['input'], 
                   'label': 'Date Processing\n- Temporal Analysis'},
        'bing_maps': {'x': 10, 'y': 7.5, 'width': box_width, 'height': box_height, 'color': colors['api'], 
                  'label': 'Bing Maps API\n- Geocoding'},
        
        # Processing Layer
        'state_terrain': {'x': 1, 'y': 5.8, 'width': box_width, 'height': box_height, 'color': colors['processing'], 
                     'label': 'State & Terrain Mapping\n- Normalize Terrain'},
        'rainfall_data': {'x': 4, 'y': 5.8, 'width': box_width, 'height': box_height, 'color': colors['processing'], 
                      'label': 'Rainfall Data Retrieval\n- Historical & Current'},
        'normalization': {'x': 7, 'y': 5.8, 'width': box_width, 'height': box_height, 'color': colors['processing'], 
                     'label': 'Data Normalization\n- StandardScaler'},
        'feature_extraction': {'x': 10, 'y': 5.8, 'width': box_width, 'height': box_height, 'color': colors['processing'], 
                          'label': 'Feature Extraction\n- ML-Ready Vectors'},
        
        # Prediction Engines
        'flood_model': {'x': 2.5, 'y': 4.1, 'width': box_width + 0.5, 'height': box_height, 'color': colors['flood'], 
                    'label': 'Flood Prediction Engine\n- TensorFlow Model\n'},
        'drought_model': {'x': 6.0, 'y': 4.1, 'width': box_width + 0.5, 'height': box_height, 'color': colors['drought'], 
                     'label': 'Drought Analysis Engine\n- Rule-Based Thresholds'},
        'landslide_model': {'x': 9.5, 'y': 4.1, 'width': box_width + 0.5, 'height': box_height, 'color': colors['landslide'], 
                       'label': 'Landslide Risk Model\n- Ensemble ML'},
        
        # Output Processing Layer
        'severity_classification': {'x': 2.5, 'y': 2.5, 'width': box_width + 0.5, 'height': box_height, 'color': colors['output'], 
                              'label': 'Severity Classification\n- Risk Levels'},
        'result_aggregation': {'x': 6.0, 'y': 2.5, 'width': box_width + 0.5, 'height': box_height, 'color': colors['output'], 
                          'label': 'Result Aggregation\n- Combine Predictions'},
        'summary_generation': {'x': 9.5, 'y': 2.5, 'width': box_width + 0.5, 'height': box_height, 'color': colors['output'], 
                          'label': 'Prediction Summary\n- User-Friendly Output'},
        
        # Final Output Layer
        'prediction_display': {'x': 4.2, 'y': 1, 'width': box_width + 0.5, 'height': box_height, 'color': colors['output'], 
                          'label': 'Climate Risk Visualization\n- Interactive UI'},
        'recommendation': {'x': 7.8, 'y': 1, 'width': box_width + 0.5, 'height': box_height, 'color': colors['output'], 
                       'label': 'Safety Recommendations\n- Actionable Advice'}
    }
    
    # Draw rounded rectangles for each component
    for name, comp in components.items():
        # Draw shadow for 3D effect (slight offset)
        if 'model' in name or 'api' in name:
            shadow = patches.FancyBboxPatch(
                (comp['x'] + 0.03, comp['y'] - 0.03), comp['width'], comp['height'],
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=corner_radius),
                facecolor='#00000022', zorder=1
            )
            ax.add_patch(shadow)
        
        # Draw the box
        box = patches.FancyBboxPatch(
            (comp['x'], comp['y']), comp['width'], comp['height'],
            boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=corner_radius),
            facecolor=comp['color'], edgecolor=colors['border'], linewidth=border_width,
            zorder=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, comp['label'],
                ha='center', va='center', color=colors['text'], fontsize=9, fontweight='bold',
                zorder=3)
    
    # Draw arrows for connections with adjusted curved values to prevent overlap
    def draw_arrow(start, end, color=colors['arrows'], width=0.005, curved=0.3, zorder=1, custom_control=None):
        """Draw a curved arrow from start to end component"""
        start_x = components[start]['x'] + components[start]['width']/2
        start_y = components[start]['y']
        end_x = components[end]['x'] + components[end]['width']/2
        end_y = components[end]['y'] + components[end]['height']
        
        # For horizontal connections, adjust x and y points
        if components[start]['y'] == components[end]['y']:
            start_x = components[start]['x'] + components[start]['width']
            start_y = components[start]['y'] + components[start]['height']/2
            end_x = components[end]['x']
            end_y = components[end]['y'] + components[end]['height']/2
        
        # Create curved path
        dx = end_x - start_x
        dy = end_y - start_y
        
        if custom_control:
            # Use custom control point
            control_point = custom_control
            path = Path([(start_x, start_y), 
                       control_point,
                       (end_x, end_y)],
                       [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        elif curved > 0:
            # For vertical arrows
            if abs(dx) < abs(dy)/2:
                control_point = ((start_x + end_x)/2, (start_y + end_y)/2 - curved * dy/2)
                path = Path([(start_x, start_y), 
                           control_point,
                           (end_x, end_y)],
                           [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            else:
                # For sideways arrows
                control_point = ((start_x + end_x)/2 + curved * dx/3, (start_y + end_y)/2)
                path = Path([(start_x, start_y), 
                           control_point,
                           (end_x, end_y)],
                           [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        else:
            # Straight line
            path = Path([(start_x, start_y), (end_x, end_y)],
                       [Path.MOVETO, Path.LINETO])
        
        # Create the patch
        arrow = patches.FancyArrowPatch(
            path=path,
            arrowstyle='-|>',
            color=color,
            lw=2,
            connectionstyle=f'arc3,rad={0.2 if curved > 0 else 0}',
            zorder=zorder
        )
        ax.add_patch(arrow)
    
    # Draw connections between components with customized curves to prevent overlap
    # Data to Input
    draw_arrow('historical_data', 'state_terrain', curved=0.4)
    draw_arrow('generated_data', 'rainfall_data', curved=0.4)
    
    # User Input Flow
    draw_arrow('user_input', 'location_input')
    draw_arrow('location_input', 'bing_maps')
    draw_arrow('location_input', 'date_input')
    
    # API & Processing - Adjust curves to prevent crossing
    draw_arrow('bing_maps', 'state_terrain', curved=0.5, 
               custom_control=(5.5, 6.8))
    draw_arrow('date_input', 'rainfall_data', curved=0.4)
    draw_arrow('state_terrain', 'rainfall_data', curved=0.2)
    draw_arrow('rainfall_data', 'normalization', curved=0.2)
    draw_arrow('normalization', 'feature_extraction', curved=0.2)
    
    # Feature processing to prediction models - Use custom control points to prevent overlap
    draw_arrow('feature_extraction', 'flood_model', curved=0.5,
               custom_control=(10, 4.8))
    draw_arrow('feature_extraction', 'drought_model', curved=0.3)
    draw_arrow('feature_extraction', 'landslide_model', curved=0.2)
    
    # Prediction to output processing
    draw_arrow('flood_model', 'severity_classification', curved=0.2)
    draw_arrow('drought_model', 'severity_classification', curved=0.4,
               custom_control=(4.8, 3.3))
    draw_arrow('landslide_model', 'severity_classification', curved=0.5,
               custom_control=(4, 3.5))
    draw_arrow('severity_classification', 'result_aggregation', curved=0.2)
    draw_arrow('result_aggregation', 'summary_generation', curved=0.2)
    
    # Final output
    draw_arrow('summary_generation', 'prediction_display', curved=0.3,
               custom_control=(7, 1.7))
    draw_arrow('summary_generation', 'recommendation', curved=0.2)
    
    # Add methodology steps labels
    ax.text(1, 11, '1. DATA SOURCES & PREPROCESSING', 
            fontsize=12, fontweight='bold', color='#1565C0')
    ax.text(1, 8.5, '2. USER INPUT & LOCATION PROCESSING', 
            fontsize=12, fontweight='bold', color='#00838F')
    ax.text(1, 6.8, '3. FEATURE ENGINEERING & DATA PREPARATION', 
            fontsize=12, fontweight='bold', color='#2E7D32')
    ax.text(1, 5, '4. CLIMATE RISK PREDICTION ENGINES', 
            fontsize=12, fontweight='bold', color='#6A1B9A')
    ax.text(1, 3.5, '5. SEVERITY CLASSIFICATION & AGGREGATION', 
            fontsize=12, fontweight='bold', color='#C62828')
    ax.text(1, 1.9, '6. RESULT VISUALIZATION & RECOMMENDATIONS', 
            fontsize=12, fontweight='bold', color='#4527A0')
    
    # Set limits and turn off axis
    ax.set_xlim(0, 15)  # Wider viewing area
    ax.set_ylim(0, 12)  # Taller viewing area
    ax.axis('off')
    
    # Add title
    #plt.title('Climate Action Prediction System Architecture', 
    #          fontsize=18, fontweight='bold', pad=20)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Add a legend for the colors at the bottom
    legend_elements = [
        patches.Patch(facecolor=colors['data'], edgecolor=colors['border'], label='Data Sources'),
        patches.Patch(facecolor=colors['input'], edgecolor=colors['border'], label='User Input Processing'),
        patches.Patch(facecolor=colors['processing'], edgecolor=colors['border'], label='Data Processing'),
        patches.Patch(facecolor=colors['flood'], edgecolor=colors['border'], label='Flood Prediction'),
        patches.Patch(facecolor=colors['drought'], edgecolor=colors['border'], label='Drought Prediction'),
        patches.Patch(facecolor=colors['landslide'], edgecolor=colors['border'], label='Landslide Prediction'),
        patches.Patch(facecolor=colors['output'], edgecolor=colors['border'], label='Output & Visualization'),
        patches.Patch(facecolor=colors['api'], edgecolor=colors['border'], label='External API Services')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(1, -0.1), ncol=4, fontsize=10, frameon=False)
    
    # Save the figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Climate Prediction System architecture diagram saved to {save_path}")
    return fig

# Generate the diagram
if __name__ == "__main__":
    fig = create_climate_prediction_architecture()

    # If running in a notebook, display the figure
    try:
        from IPython.display import display
        display(fig)
    except:
        pass