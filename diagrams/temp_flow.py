import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

def create_work_flow_diagram(completed_tasks, title="Work Completed Flow Diagram", 
                            save_path="work_flow_diagram.png", dpi=300):
    """
    Creates a boxy, architectural-style visualization of completed work items with interconnections.
    
    Args:
        completed_tasks: A dictionary where keys are task names and values are tuples of 
                        (color, description).
        title: Title for the flow diagram.
        save_path: Path to save the image file.
        dpi: Resolution of the output image.
    """
    # Set up the figure
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#FAFAFA')
    
    # Define the flow connections between tasks
    # Format: (from_task_index, to_task_index)
    connections = [
        (0, 2),  # EDA → Dataset Finalization
        (1, 2),  # Literature Review → Dataset Finalization
        (2, 3),  # Dataset Finalization → Code Development
        (3, 4),  # Code Development → Validation and Improvement
        (4, 5),  # Validation and Improvement → Research Paper
    ]
    
    # Calculate grid layout (roughly 3x2 grid)
    task_names = list(completed_tasks.keys())
    num_tasks = len(task_names)
    
    cols = 3
    rows = (num_tasks + cols - 1) // cols
    
    # Calculate positions for each box
    positions = {}
    max_width = 2.5  # Maximum width for boxes
    max_height = 1.0  # Maximum height for boxes
    spacing_x = 3.5  # Horizontal spacing between boxes
    spacing_y = 2.3  # Vertical spacing between boxes
    
    # Calculate grid positions
    for i, task in enumerate(task_names):
        row = i // cols
        col = i % cols
        
        # Center-align boxes in the grid
        center_x = col * spacing_x + max_width/2
        center_y = (rows - row - 1) * spacing_y + max_height/2
        
        positions[task] = (center_x, center_y)
    
    # Draw boxes for each task
    box_width = max_width * 0.9
    box_height = max_height * 0.8
    corner_radius = 0.2
    
    for i, (task, (color, description)) in enumerate(completed_tasks.items()):
        x, y = positions[task]
        
        # Calculate box position (centered at x,y)
        box_x = x - box_width/2
        box_y = y - box_height/2
        
        # Add shadow for 3D effect
        shadow = patches.FancyBboxPatch(
            (box_x + 0.08, box_y - 0.08), 
            box_width, box_height,
            boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=corner_radius),
            facecolor='#33333333', 
            linewidth=0,
            zorder=1
        )
        ax.add_patch(shadow)
        
        # Draw the box
        task_box = patches.FancyBboxPatch(
            (box_x, box_y), 
            box_width, box_height,
            boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=corner_radius),
            facecolor=color, 
            alpha=0.85,
            edgecolor='#333333',
            linewidth=2,
            zorder=2
        )
        ax.add_patch(task_box)
        
        # Add task name
        ax.text(x, y, task, 
                ha='center', va='center', 
                color='white', fontweight='bold', fontsize=11, zorder=3)
        
        # Add description below the task name if provided
        if description:
            desc_y = y - 0.2
            ax.text(x, desc_y, description, 
                    ha='center', va='center', 
                    color='white', fontsize=8, fontstyle='italic', zorder=3)
    
    # Draw connection arrows between tasks
    for from_idx, to_idx in connections:
        from_task = task_names[from_idx]
        to_task = task_names[to_idx]
        
        from_x, from_y = positions[from_task]
        to_x, to_y = positions[to_task]
        
        # Calculate start/end positions on the box edges
        # Determine if the connection is horizontal, vertical, or diagonal
        dx = to_x - from_x
        dy = to_y - from_y
        
        # Determine start point (from box edge)
        if abs(dx) > abs(dy):
            # Horizontal connection
            start_x = from_x + box_width/2 * np.sign(dx)
            start_y = from_y
        else:
            # Vertical connection
            start_x = from_x
            start_y = from_y + box_height/2 * np.sign(dy)
        
        # Determine end point (to box edge)
        if abs(dx) > abs(dy):
            # Horizontal connection
            end_x = to_x - box_width/2 * np.sign(dx)
            end_y = to_y
        else:
            # Vertical connection
            end_x = to_x
            end_y = to_y - box_height/2 * np.sign(dy)
        
        # Calculate control points for curved arrow
        if abs(dx) > abs(dy):
            # For horizontal connections
            control_x = (start_x + end_x) / 2
            control_y = (start_y + end_y) / 2 + 0.4 * np.sign(dy) if dy != 0 else (start_y + end_y) / 2
        else:
            # For vertical connections
            control_x = (start_x + end_x) / 2 + 0.4 * np.sign(dx) if dx != 0 else (start_x + end_x) / 2
            control_y = (start_y + end_y) / 2
        
        # Create curved path
        arrow_path = Path(
            [(start_x, start_y), 
             (control_x, control_y),
             (end_x, end_y)],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        )
        
        # Create the arrow patch
        arrow = patches.FancyArrowPatch(
            path=arrow_path,
            arrowstyle='-|>',
            color='#555555',
            linewidth=2,
            connectionstyle='arc3,rad=0.1',
            zorder=1,
            shrinkA=0,
            shrinkB=0
        )
        ax.add_patch(arrow)
    
    # Set up the plot
    ax.set_xlim(-max_width/2, cols * spacing_x)
    ax.set_ylim(-max_height/2, rows * spacing_y + max_height)
    ax.axis('off')
    
    # Add title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add completion indicator
    completion_text = "ALL TASKS COMPLETED"
    ax.text(cols * spacing_x / 2, -0.8, completion_text,
            ha='center', va='center', color='#2E7D32',
            fontweight='bold', fontsize=14,
            bbox=dict(facecolor='#E8F5E9', edgecolor='#2E7D32', 
                     boxstyle='round,pad=0.5', alpha=0.9))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Work flow diagram saved to {save_path}")
    return fig

# Create the diagram
if __name__ == "__main__":
    completed_tasks = {
        'Exploratory Data Analysis': ('#4E79A7', 'Data exploration & insights'),
        'Literature Review': ('#F28E2B', 'Research background'),
        'Dataset Finalization': ('#E15759', 'Final data selection'),
        'Code Development': ('#76B7B2', 'ML model implementation'),
        'Validation and Improvement': ('#59A14F', 'Testing & refinement'),
        'Research Paper': ('#EDC948', 'Documentation & findings')
    }
    
    create_work_flow_diagram(completed_tasks, 
                            title="Climate Prediction System: Research Workflow",
                            save_path="research_workflow.png")