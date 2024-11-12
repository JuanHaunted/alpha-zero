import matplotlib.pyplot as plt

# Data for ConnectFour Search Time vs Current Turn
connect_four_times = [
    (7.677586078643799, 1),
    (7.075190782546997, 2),
    (8.050450801849365, 3),
    (9.244024753570557, 4),
    (6.7563958168029785, 5),
    (5.510528564453125, 6),
    (7.849049091339111, 7),
    (6.7762908935546875, 8),
    (6.205289125442505, 9),
    (3.6819562911987305, 10),
    (0.16375422477722168, 11)
]

# Unpacking data
time, turn = zip(*connect_four_times)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(turn, time, marker='o', color='b', linestyle='-', linewidth=2, markersize=6, markerfacecolor='orange')

# Adding title and labels
plt.title('ConnectFour Search Time vs Current Turn', fontsize=16, fontweight='bold')
plt.xlabel('Current Turn', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)

# Adding grid and customizing plot aesthetics
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Display plot
plt.show()