import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import torch
from torchviz import make_dot
import os


def visualize_grad_graph(model, loss, output_dir="./graphs"):
    """
    Visualizes and saves the computational graph of a PyTorch model.

    Args:
        model: PyTorch model
        loss: Input tensor to the model
        output_dir: Directory to save the graph image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Forward pass
    # output = model(input_tensor)
    # Create dot graph
    dot = make_dot(loss, params=dict(model.named_parameters()))

    # Customize graph appearance
    # dot.attr(rankdir='LR')  # Left to right layout
    # dot.attr('node', shape='record')

    # Save graph in different formats
    dot.render(os.path.join(output_dir, "model_graph"), format="png", cleanup=True)
    print(f"Graph saved to {output_dir}/model_graph.png")