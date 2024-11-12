import matplotlib.pyplot as plt

# Data
times = [
    (50, 0.348),  
    (100, 0.568),
    (150, 0.834),
    (200, 1.108),
    (250, 1.618),
    (300, 1.218),
    (350, 1.464),
    (400, 1.190),
    (450, 1.689),
    (500, 2.604),
    (550, 1.571),
    (600, 1.595)
]

# Unpacking data
x, y = zip(*times)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', color='b', linestyle='-', linewidth=2, markersize=6, markerfacecolor='orange')

# Adding title and labels
plt.title('TicTacToe AlphaMCTS Search Time', fontsize=16, fontweight='bold')
plt.xlabel('Number of Searches', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)

# Adding grid and customizing plot aesthetics
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Display plot
plt.show()