import matplotlib.pyplot as plt

datas = [
    {
        "label": "pgvector (hnsw)",
        "data": [
            (0.990, 290.740),
            (0.976, 452.143),
            (0.950, 625.835),
            (0.891, 824.790)
        ]
    },
    {
        "label": "pgvector(hnsw 2)",
        "data": [
            (0.989, 232.261),
            (0.981, 332.755),
            (0.964, 410.807),
            (0.939, 498.576)
        ]
    }
]

# List of different markers to cycle through
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']

# Iterate through each dataset in 'datas' and plot lines and scatter points
for i, data in enumerate(datas):
    x_values, y_values = zip(*data["data"])
    # Check if the label contains 'z'
    if 'z' in data["label"]:
        linestyle = '--'  # Use dashed line if label contains 'z'
    else:
        linestyle = '-'  # Use solid line for others
    # Plot the line with label
    plt.plot(x_values, y_values, linestyle=linestyle, linewidth=1, label=data["label"])
    # Plot the points without additional label to avoid double labeling
    plt.scatter(x_values, y_values, marker=markers[i % len(markers)])

# Update axis labels
plt.title('Test Result')
plt.xlabel('Recall')
plt.ylabel('QPS')
plt.legend()

# Add grid lines
plt.grid(True)

# Show the plot
plt.show()
