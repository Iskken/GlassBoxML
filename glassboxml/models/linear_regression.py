import math
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.b = 0
        self.learning_rate = 0
        self.loss = 0
        self.epsilon = 1e-6
        self.losses = []
        pass

    def compute_b(self, computed_pts, X, y):
        # av_x = 0
        # av_y = 0
        # for point in data:
        #     av_x += point[0]
        #     av_y += point[1]

        # av_x = av_x / len(data)
        # av_y = av_y / len(data)

        # b = av_y - self.w * av_x
        # return b

        b = np.mean(y) - self.w @ np.mean(X)
        return b

    def compute_weight(self, X,y, pred_y):
        # av_x = 0
        # av_y = 0
        # for point in data:
        #     av_x += point[0]
        #     av_y += point[1]

        # av_x = av_x / len(data)
        # av_y = av_y / len(data)

        # numerator = 0
        # denumerator = 0
        # for point in computed_pts:
        #     numerator += (point[0] - av_x) * (point[1] - av_y)
        #     denumerator += math.pow(point[0] - av_x,2)

        # weight = numerator / denumerator
        # return weight 
        # X is n by 1, y is n by 1, so (X - np.mean(X)).T @ (y - np.mean(y)) is 1 by 1,
        # and np.sum((X - np.mean(X))**2) is a scalar, so the result is a scalar
        w = (X - np.mean(X)).T @ (y - np.mean(y)) / np.sum((X - np.mean(X))**2)
        return w

    def predict(self, X):
        # new_data = []
        # for point in data:
        #     new_data.append((point[0], self.w * point[0] + self.b))

        # return new_data
        pred_y = X @ self.w + self.b
        return pred_y
    
    def fit_closed_form(self, X, y):
        av_x = np.mean(X)
        av_y = np.mean(y)

        numerator = (X - av_x).T @ (y - av_y)
        denominator = np.sum((X - av_x)**2)

        self.w = numerator / denominator
        self.b = av_y - self.w * av_x

    def fit_gradient_descent(self, X, y, epochs, learning_rate):
        # Initialize weights and bias
        self.w = np.zeros(len(X[0]))
        self.b = 0

        for epoch in range(epochs):
            computed_pts = self.compute_points(data)
            self.losses.append(self.compute_loss(computed_pts=computed_pts, data=data))
            
            # Calculate gradient for weight
            sum = 0
            for i in range(len(computed_pts)):
                sum += computed_pts[i][0] * (data[i][1] - computed_pts[i][1])
            dw = (-2) / len(data) * sum

            # Calculate gradient for b
            sum = 0
            for i in range(len(computed_pts)):
                sum += data[i][1] - computed_pts[i][1]
            db = (-2) / len(data) * sum

            if abs(dw) < self.epsilon and abs(db) < self.epsilon:
                print("Converged at epoch", epoch)
                break

            #update weight and b
            self.w = self.w - learning_rate*dw
            self.b = self.b - learning_rate*db

            if (epoch % 20 == 0):
                print("Updated weight: ", self.w)
                print("Updated b:", self.b, "\n")

    def plot(self, data):
        # Separate original data
        x_vals = np.array([point[0] for point in data])
        y_vals = np.array([point[1] for point in data])

        # Create smooth line range
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        y_line = self.w * x_line + self.b

        # Plot data points
        plt.scatter(x_vals, y_vals)

        # Plot regression line
        plt.plot(x_line, y_line)

        # Labels
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Linear Regression | w={self.w:.3f}, b={self.b:.3f}")

        plt.show()

    def compute_loss(self, computed_pts, data):
        loss = 0
        for i in range(len(computed_pts)):
            loss += math.pow(data[i][1] - computed_pts[i][1], 2)
        
        return loss