import numpy as np

def piecewise_alternating(x, n_pieces=4, x_range=(-2, 2), a=1.0):
    """
    Piecewise function of the form (a + (-1)^k/n) * x where k is the piece index.
    
    This implements your original idea: (a + (-1)^k/n) * x
    
    Parameters:
    - x: input values
    - n_pieces: number of pieces to divide the range into
    - x_range: tuple of (min_x, max_x) defining the domain
    - a: base parameter for the function
    
    Returns:
    - y: output values with alternating coefficients
    """
    x_min, x_max = x_range
    piece_width = (x_max - x_min) / n_pieces
    
    # Determine which piece each x belongs to
    piece_indices = np.floor((x - x_min) / piece_width).astype(int)
    piece_indices = np.clip(piece_indices, 0, n_pieces - 1)
    
    # Calculate the coefficient for each piece: (a + (-1)^k/n)
    coefficients = a + ((-1) ** piece_indices) / n_pieces
    
    return coefficients * x

def piecewise_linear_alternating(x, n_pieces=4, x_range=(-2, 2), a=1.0):
    """
    Alternative piecewise function with linear segments and alternating slopes.
    Each piece has slope alternating between positive and negative values.
    This ensures continuity between pieces.
    
    Parameters:
    - x: input values
    - n_pieces: number of pieces to divide the range into
    - x_range: tuple of (min_x, max_x) defining the domain
    - a: base parameter for the function
    
    Returns:
    - y: output values with continuous alternating linear segments
    """
    x_min, x_max = x_range
    piece_width = (x_max - x_min) / n_pieces
    
    # Determine which piece each x belongs to
    piece_indices = np.floor((x - x_min) / piece_width).astype(int)
    piece_indices = np.clip(piece_indices, 0, n_pieces - 1)
    
    # Calculate slopes for each piece (alternating positive/negative)
    slopes = a * ((-1) ** piece_indices)
    
    # Calculate y-intercept for each piece to ensure continuity
    y_intercepts = np.zeros_like(piece_indices, dtype=float)
    for i in range(1, n_pieces):
        # Calculate where the previous piece ends
        prev_end_x = x_min + i * piece_width
        prev_slope = a * ((-1) ** (i - 1))
        prev_end_y = prev_slope * prev_end_x + y_intercepts[piece_indices == i - 1][0] if np.any(piece_indices == i - 1) else 0
        
        # Set current piece to start at the same y-value
        current_slope = a * ((-1) ** i)
        y_intercepts[piece_indices == i] = prev_end_y - current_slope * prev_end_x
    
    return slopes * x + y_intercepts

def piecewise_simple_alternating(x, n_pieces=4, x_range=(-2, 2), a=1.0):
    """
    Simplified piecewise function with alternating coefficients.
    Each piece multiplies x by an alternating coefficient: a, -a, a, -a, ...
    
    Parameters:
    - x: input values
    - n_pieces: number of pieces to divide the range into
    - x_range: tuple of (min_x, max_x) defining the domain
    - a: base parameter for the function
    
    Returns:
    - y: output values with simple alternating coefficients
    """
    x_min, x_max = x_range
    piece_width = (x_max - x_min) / n_pieces
    
    # Determine which piece each x belongs to
    piece_indices = np.floor((x - x_min) / piece_width).astype(int)
    piece_indices = np.clip(piece_indices, 0, n_pieces - 1)
    
    # Simple alternating coefficients: a, -a, a, -a, ...
    coefficients = a * ((-1) ** piece_indices)
    
    return coefficients * x

# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create test data
    x = np.linspace(-2, 2, 1000)
    
    # Test different piecewise functions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original function: (a + (-1)^k/n) * x
    y1 = piecewise_alternating(x, n_pieces=4, a=1.0)
    axes[0].plot(x, y1, 'b-', linewidth=2)
    axes[0].set_title('Original: (a + (-1)^k/n) * x')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True)
    
    # Continuous linear segments
    y2 = piecewise_linear_alternating(x, n_pieces=4, a=1.0)
    axes[1].plot(x, y2, 'r-', linewidth=2)
    axes[1].set_title('Continuous: Alternating Linear Segments')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True)
    
    # Simple alternating
    y3 = piecewise_simple_alternating(x, n_pieces=4, a=1.0)
    axes[2].plot(x, y3, 'g-', linewidth=2)
    axes[2].set_title('Simple: a, -a, a, -a, ...')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Piecewise function definitions:")
    print("1. piecewise_alternating: Your original idea (a + (-1)^k/n) * x")
    print("2. piecewise_linear_alternating: Continuous alternating linear segments")
    print("3. piecewise_simple_alternating: Simple alternating coefficients a, -a, a, -a, ...") 