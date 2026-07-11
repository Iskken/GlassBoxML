from experiments.decision_tree.basic_dTree import DecisionTree

'''
Purpose: Test the basic implementation of decision tree
'''

X1 = [[1], [2], [3], [10], [11], [12]]
y1 = [0, 0, 0, 1, 1, 1]

X2 = [
    [1, 1], [2, 1], [1, 2],
    [8, 8], [9, 8], [8, 9]
]
y2 = [0, 0, 0, 1, 1, 1]

X3 = [[1], [2], [3], [4], [5]]
y3 = [0, 0, 1, 0, 1]

model = DecisionTree()

node = model.build_tree(X1, y1, 0, 3)

pred = model.predict_sample([[6]], node)

print(pred)