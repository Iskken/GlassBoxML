import math
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0
        self.learning_rate = 0
        self.loss = 0
        self.epsilon = 1e-6
        pass

    def compute_b(self, computed_pts, data):
        av_x = 0
        av_y = 0
        for point in data:
            av_x += point[0]
            av_y += point[1]

        av_x = av_x / len(data)
        av_y = av_y / len(data)

        b = av_y - self.w * av_x
        return b

    def compute_weight(self, computed_pts, data):
        av_x = 0
        av_y = 0
        for point in data:
            av_x += point[0]
            av_y += point[1]

        av_x = av_x / len(data)
        av_y = av_y / len(data)

        numerator = 0
        denumerator = 0
        for point in computed_pts:
            numerator += (point[0] - av_x) * (point[1] - av_y)
            denumerator += math.pow(point[0] - av_x,2)

        weight = numerator / denumerator
        return weight 

    def compute_points(self, data):
        new_data = []
        for point in data:
            new_data.append((point[0], self.w * point[0] + self.b))

        return new_data
    
    def fit_closed_form(self, data):
        av_x = sum([p[0] for p in data]) / len(data)
        av_y = sum([p[1] for p in data]) / len(data)

        numerator = sum((p[0] - av_x)*(p[1] - av_y) for p in data)
        denominator = sum((p[0] - av_x)**2 for p in data)

        self.w = numerator / denominator
        self.b = av_y - self.w * av_x

    def fit_gradient_descent(self, data, epochs, learning_rate):
        for epoch in range(epochs):
            computed_pts = self.compute_points(data)

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