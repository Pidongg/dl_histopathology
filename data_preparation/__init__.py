import numpy as np
import matplotlib.pyplot as plt

def plot_losses():
    x = np.linspace(-3, 3, 1000)
    
    # Calculate losses
    huber = np.where(np.abs(x) <= 1, 
                     0.5 * x**2, 
                     np.abs(x) - 0.5)
    
    smooth_l1 = np.where(np.abs(x) <= 1, 
                        0.5 * x**2, 
                        np.abs(x) - 0.5)
    
    plt.plot(x, huber, label='Huber Loss')
    plt.plot(x, smooth_l1, '--', label='Smooth L1 Loss')
    plt.grid(True)
    plt.legend()
    plt.title('Huber Loss vs Smooth L1 Loss')
    plt.xlabel('Error')
    plt.ylabel('Loss')
    plt.show()

plot_losses()
