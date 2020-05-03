import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from other_metrics import *

# Example 1:
print("\n\nExemple 1:")
y_hat = np.array(   [1, 1, 0, 1, 0, 0, 1, 1])
y = np.array(       [1, 0, 0, 1, 0, 1, 0, 0])

print("\nAccuracy:")
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))
## sklearn implementation 
print(accuracy_score(y, y_hat))
## Output: 0.5

print("\nPrecision:")
# Precision
## your implementation
print(precision_score_(y, y_hat))
## sklearn implementation
print(precision_score(y, y_hat))
## Output: 0.4

print("\nRecall:")
# Recall
## your implementation 
print(recall_score_(y, y_hat))
## sklearn implementation
print(recall_score(y, y_hat))
## Output: 0.6666666666666666

print("\nF1-score:")
# F1-score
## your implementation
print(f1_score_(y, y_hat))
## sklearn implementation
print(f1_score(y, y_hat))
## Output: 0.5


# Example 2:
print("\n\nExemple 2:")
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

print("\nAccuracy:")
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))
## sklearn implementation
print(accuracy_score(y, y_hat))
## Output: 0.625

print("\nPrecision:")
# Precision
## your implementation
print(precision_score_(y, y_hat, pos_label='dog'))
## sklearn implementation
print(precision_score(y, y_hat, pos_label='dog'))
## Output: 0.6

print("\nRecall:")
# Recall
## your implementation
print(recall_score_(y, y_hat, pos_label='dog'))
## sklearn implementation
print(recall_score(y, y_hat, pos_label='dog'))
## Output: 0.75

print("\nF1-score:")
# F1-score
## your implementation
print(f1_score_(y, y_hat, pos_label='dog'))
## sklearn implementation
print(f1_score(y, y_hat, pos_label='dog'))
## Output: 0.6666666666666665


# Example 3:
print("\n\nExemple 3:")
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

print("\nAccuracy:")
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))
## sklearn implementation
print(accuracy_score(y, y_hat))
## Output: 0.625


print("\nPrecision:")
# Precision
## your implementation
print(precision_score_(y, y_hat, pos_label='norminet'))
## sklearn implementation
print(precision_score(y, y_hat, pos_label='norminet'))
## Output: 0.6666666666666666

print("\nRecall:")
# Recall
## your implementation
print(recall_score_(y, y_hat, pos_label='norminet'))
## sklearn implementation
print(recall_score(y, y_hat, pos_label='norminet'))
## Output: 0.5

print("\nF1-score:")
# F1-score
## your implementation
print(f1_score_(y, y_hat, pos_label='norminet'))
## sklearn implementation
print(f1_score(y, y_hat, pos_label='norminet'))
## Output: 0.5714285714285715