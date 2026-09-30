import matplotlib.pyplot as plt
import seaborn as sns


sns.set_context("poster")
sns.set_style("white")


plt.rcParams.update({
    'figure.figsize': (10, 6),          # Figure size in inches
    'axes.titlesize': 24,               # Title font size
    'axes.labelsize': 20,               # Axis label font size
    'xtick.labelsize': 18,              # X tick label font size
    'ytick.labelsize': 18,              # Y tick label font size
    'legend.fontsize': 18,              # Legend font size
    'lines.linewidth': 2.5,             # Line width
    'lines.markersize': 10,             # Marker size
    'axes.linewidth': 2,                # Axis line width
    'grid.linewidth': 1.5,              # Grid line width
    'font.family': 'sans-serif',        # Font family
    'font.sans-serif': ['Arial'],       # Font type
    'legend.frameon': True,             # Draw legend box
    'legend.loc': 'best',               # Best location for legend
    'legend.title_fontsize': 18,        # Legend title font size
    'grid.color': '0.8',                # Grid color
    'xtick.major.size': 10,             # Major tick size for x-axis
    'xtick.minor.size': 6,              # Minor tick size for x-axis
    'ytick.major.size': 10,             # Major tick size for y-axis
    'ytick.minor.size': 6,              # Minor tick size for y-axis
    'xtick.direction': 'in',            # Tick direction for x-axis
    'ytick.direction': 'in',            # Tick direction for y-axis
    'axes.grid': True,                  # Show grid
    'grid.alpha': 0.7,                  # Grid transparency
    'xtick.major.pad': 10,              # Padding between x ticks and x tick labels
    'ytick.major.pad': 10               # Padding between y ticks and y tick labels
})


colors = {
    'white': '#FFFFFF',
    'blue': '#2B4B83', # '#2B4B83',
    'red': '#C41425',
    'alt_red': '#ae2a37',
    'orange': '#F78C1E',
    'lightorange': '#FFA07A',
    'alt_lightorange': '#eea78b',
    'cyan': '#88D2D6',
    'pink': '#F492B3',
    'black': '#1B1B1B',
    'gray': '#758D99',
    'lightgray': '#a7b6be',
    'brown': '#8F6F5D'
}


grayscale = {
    'early': '#ffffff',
    'late': '#a0a0a0',
    'all': '#787878',
    'markers': '#555555',
    'edges': '#222222',
}


palette_early_late = [colors['lightorange'], colors['red']]
palette_methods_4 = [colors['blue'], colors['cyan'], colors['pink'], colors['lightorange']]
palette_methods_profshuf = [colors['blue'], colors['cyan']]
palette_all = [colors['gray']]


color_early = colors['lightorange']
color_late = colors['red']


color_lines = grayscale["markers"]
color_markers = grayscale["markers"]
color_edges = grayscale["markers"]