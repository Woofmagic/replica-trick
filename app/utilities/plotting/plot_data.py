# External Library | Matplotlib
import matplotlib.pyplot as plt
import sympy as sp

from app.utilities.plotting.plot_customizer import PlotCustomizer

def construct_plot(x_data, y_data, analytical_form):
    
    # (1): Set up the Figure instance
    figure_instance = plt.figure(figsize = (18, 6))

    # (2): Add an Axes Object:
    axis_instance = figure_instance.add_subplot(1, 1, 1)
    
    # (3): Customize the Axes Object:
    plot_customization = PlotCustomizer(
        axis_instance,
        title = r"Function: $f(x) = {}$".format(sp.latex(analytical_form)),
        xlabel = r"$x$",
        ylabel = r"$f(x)$")
    
    # (4): Add data to the Axes Object:
    plot_customization.add_scatter_plot(
        x_data, 
        y_data,  
        color = 'black')
    
    plt.show()