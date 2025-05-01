import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_interactive_confusion_matrix(model, data_yaml, class_names, save_path, iou_thresh=0.5, conf_range=None):
    """Create and save interactive confusion matrix plot as HTML
    
    Visualizes all specified combinations of per-class confidence thresholds.
    
    Args:
        model: YOLO model
        data_yaml: Path to data yaml file
        class_names: Dictionary of class names
        save_path: Path to save HTML output
        iou_thresh: IoU threshold for evaluation
        conf_range: List or array of confidence thresholds to test (default: np.arange(0.01, 0.4, 0.05))
    """
    if conf_range is None:
        conf_range = np.arange(0.01, 0.31, 0.05)  # Default range for thresholds. not 0.31
    
    num_classes = len(class_names)
    
    # Initialize lists to store results
    all_results = []
    computed_results = {}  # Cache to avoid repeating computations
    
    # Generate a more limited set of threshold combinations
    # Instead of generating all possible combinations (exponential),
    # we'll vary one class threshold at a time (linear)
    default_threshold = 0.01  # Use the first value as default
    threshold_combinations = []
    
    # First, add combinations where all classes use the same threshold
    for t in conf_range:
        threshold_combinations.append([t] * num_classes)
    
    # Then, for each class, vary its threshold while keeping others at default
    for class_idx in range(num_classes):
        for t in conf_range:
            if t == default_threshold:
                continue  # Skip if it's the default (already covered above)
            thresholds = [default_threshold] * num_classes
            thresholds[class_idx] = t
            threshold_combinations.append(thresholds)
        
    # Grid search for all threshold combinations
    print(f"Starting visualization with {len(threshold_combinations)} threshold combinations")
    
    
    # Test all combinations
    for i, thresholds in enumerate(threshold_combinations):
        if i % 10 == 0:
            print(f"Testing combination {i+1}/{len(threshold_combinations)}")
        
        # Convert thresholds to a hashable format
        thresholds_key = tuple(thresholds)
        
        # Skip if already computed
        if thresholds_key in computed_results:
            result = computed_results[thresholds_key]
            all_results.append(result)
            continue
        
        # Run evaluation
        metrics = model.val(
            data=data_yaml, 
            conf=min(thresholds),  # Use minimum threshold as base
            iou=iou_thresh,
            class_conf_thresholds=list(thresholds)
        )
        
        # Get confusion matrix
        matrix = metrics.confusion_matrix.matrix
        
        # Normalize matrix
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        normalized_matrix = matrix / col_sums * 100  # Convert to percentages
        
        # Calculate diagonal score (sum of diagonal elements in normalized matrix)
        # This represents the sum of true positive rates for each class
        diag_score = np.trace(normalized_matrix[:num_classes, :num_classes])
        
        # Store the result
        result = {
            'thresholds': thresholds,
            'matrix': normalized_matrix,
            'map': metrics.box.map,
            'map50': metrics.box.map50,
            'diag_score': diag_score
        }
        
        # Cache the result
        computed_results[thresholds_key] = result
        all_results.append(result)
    
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
    
    # Initial confusion matrix (first one)
    initial_matrix = all_results[0]['matrix']
    
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
    
    # Add metrics plot for all configurations
    configurations = list(range(len(all_results)))
    maps = [r['map'] for r in all_results]
    map50s = [r['map50'] for r in all_results]
    
    fig.add_trace(
        go.Scatter(x=configurations, y=maps, name='mAP50-95',
                  mode='lines+markers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=configurations, y=map50s, name='mAP50',
                  mode='lines+markers'),
        row=2, col=1
    )
    
    # Add slider
    steps = []
    for i, result in enumerate(all_results):
        thresholds = result['thresholds']
        threshold_text = ", ".join([f"{class_names[j]}: {t:.2f}" for j, t in enumerate(thresholds)])
        diag_score = result['diag_score']
        step = dict(
            method="update",
            args=[{"z": [result['matrix']],
                  "text": [result['matrix']]},
                 {"title": f"Confusion Matrix<br>Thresholds: {threshold_text}<br>mAP50: {result['map50']:.4f}, mAP50-95: {result['map']:.4f}"}],
            label=f"Config {i+1}"
        )
        steps.append(step)
    
    # Sort results by diagonal score for better comparison
    all_results.sort(key=lambda x: x['diag_score'], reverse=True)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Configuration: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # Update layout
    fig.update_layout(
        sliders=sliders,
        height=900,
        title_text=f"Interactive Confusion Matrix<br>",
        showlegend=True
    )
    
    # Add annotation for thresholds
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text=steps[0]["args"][1]["title"].split("<br>")[1],
        showarrow=False,
        font=dict(size=14)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="True", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Configuration", row=2, col=1)
    fig.update_yaxes(title_text="Metric Value", row=2, col=1)
    
    # Save as HTML file
    fig.write_html(save_path)
    print(f"Interactive plot saved to {save_path}")
    print(f"Tested {len(all_results)} different threshold configurations")
