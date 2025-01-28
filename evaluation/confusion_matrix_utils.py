from ultralytics.utils import LOGGER
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_confusion_matrix(confusion_matrix, class_names):
    """Plot confusion matrix using seaborn"""
    # Get the matrix data and resize it to match the number of classes
    array = confusion_matrix.matrix
    n_classes = len(class_names)
    array = array[:n_classes, :n_classes]  # Take only the relevant classes

    df_cm = pd.DataFrame(array, index=class_names, columns=class_names)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def save_interactive_confusion_matrix(model, data_yaml, class_names, save_path, iou_thresh=0.5, conf_thresholds=None):
    """Create and save interactive confusion matrix plot as HTML"""
    if conf_thresholds is None:
        conf_thresholds = np.arange(0.0, 0.5, 0.05)  # From 0.1 to 0.9 in steps of 0.05
    # Initialize lists to store metrics
    maps = []
    map50s = []
    matrices = []
    
    # Collect data for all thresholds
    for conf in conf_thresholds:
        metrics = model.val(data=data_yaml, conf=float(conf), iou=iou_thresh)
        
        # Store metrics
        maps.append(metrics.box.map)
        map50s.append(metrics.box.map50)
        
        # Get confusion matrix
        array = metrics.confusion_matrix.matrix
        matrices.append(array)
    
    # Normalize matrices
    normalized_matrices = []
    for matrix in matrices:
        # Normalize by dividing each row by its sum (if sum is not zero)
        col_sums = matrix.sum(axis=0, keepdims=True)
        # Avoid division by zero by setting zero sums to 1
        col_sums[col_sums == 0] = 1
        normalized_matrix = matrix / col_sums * 100  # Convert to percentages
        normalized_matrices.append(normalized_matrix)
    
    # Create interactive plot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Confusion Matrix', 'Performance Metrics'),
        specs=[[{"type": "heatmap"}],
               [{"type": "scatter"}]]
    )
    
    # Convert class_names dictionary to ordered list
    class_list = [class_names[i] for i in range(len(class_names))] + ['background']
    
    # Initial confusion matrix
    initial_matrix = normalized_matrices[0]
    
    heatmap = go.Heatmap(
        z=initial_matrix,
        x=class_list,
        y=class_list,
        colorscale='Blues',
        text=initial_matrix,
        texttemplate='%{text:.1f}%',
        textfont={"size": 12},
        showscale=True,
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # Add metrics plot
    fig.add_trace(
        go.Scatter(x=conf_thresholds, y=maps, name='mAP50-95',
                  mode='lines+markers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=conf_thresholds, y=map50s, name='mAP50',
                  mode='lines+markers'),
        row=2, col=1
    )
    
    # Add slider
    steps = []
    for i, conf in enumerate(conf_thresholds):
        step = dict(
            method="update",
            args=[{"z": [normalized_matrices[i]],
                  "text": [normalized_matrices[i]]},
                 {"title": f"Confusion Matrix (Confidence > {conf:.2f})"}],
            label=f"{conf:.2f}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Confidence Threshold: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # Update layout
    fig.update_layout(
        sliders=sliders,
        height=900,
        title_text="Interactive Confusion Matrix with Metrics",
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="True", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Threshold", row=2, col=1)
    fig.update_yaxes(title_text="Metric Value", row=2, col=1)
    
    # Save as HTML file
    fig.write_html(save_path)
    LOGGER.info(f"Interactive plot saved to {save_path}")