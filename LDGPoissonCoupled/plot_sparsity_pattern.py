import matplotlib.pyplot as plt
import re

# Load the data
with open('sparsity_pattern21.txt', 'r') as f:
#with open('nonzero_pattern.txt', 'r') as f:
    data = f.read()

# Extract (row, column) pairs
matches = re.findall(r'\((\d+),(\d+)\)', data)
rows = [int(r) for r, c in matches]
cols = [int(c) for r, c in matches]


#with open('sparsity_pattern2.txt', 'r') as f:
with open('nonzero_pattern1.txt', 'r') as f:
    data = f.read()

# Extract (row, column) pairs
matches = re.findall(r'\((\d+),(\d+)\)', data)
rows1 = [int(r) for r, c in matches]
cols1 = [int(c) for r, c in matches]

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(cols, rows, marker='s', s=30, edgecolors='k', facecolors='none')
plt.scatter(cols1, rows1, marker='s', s=30, edgecolors='k')
plt.gca().invert_yaxis()
plt.xlabel("Column Index - trial")
plt.ylabel("Row Index -test")
plt.title("Sparsity Pattern of Matrix")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
