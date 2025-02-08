import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def tanh_function(h, d, p):
    return np.tanh(h**(p) / np.log((d**(p-1)+1)))

h = np.linspace(0, 1, 100)
d = np.linspace(0.01, 1, 100)  
H, D = np.meshgrid(h, d)  

initial_p = 1

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def plot_surface(p):
    Z = tanh_function(H, D, p)
    ax.clear()
    surface = ax.plot_surface(H, D, Z, cmap='viridis')

    ax.set_xlabel('happiness difference')
    ax.set_ylabel('inversion distance')
    ax.set_zlabel('risk')
    ax.set_title(f'3D Plot of Risk with p={p}')

    plt.draw()

ax_p = plt.axes([0.2, 0.02, 0.65, 0.03])  
p_slider = Slider(ax_p, 'p', 0.1, 10, valinit=initial_p, valstep=0.1)

# Update function for the slider
def update(val):
    p = p_slider.val
    plot_surface(p)

p_slider.on_changed(update)

plot_surface(initial_p)

plt.show()
