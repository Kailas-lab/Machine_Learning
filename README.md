# Machine_Learning
Data Science

# 1. Correlation Matrix (data.corr())
The corr() function computes the Pearson correlation coefficient for all numerical columns in the DataFrame.

What is Correlation?
Correlation quantifies the strength and direction of a linear relationship between two variables.
Values range from -1 to 1:
1: Perfect positive correlation (as one variable increases, the other also increases).
-1: Perfect negative correlation (as one variable increases, the other decreases).
0: No linear correlation.

# 2. Heatmap (sns.heatmap)
A heatmap is a graphical representation of the correlation matrix, making it easier to interpret.

How the heatmap works:
Data Preparation:

The correlation = data.corr() result is passed to the sns.heatmap() function.
Color Map (cmap):

The color (e.g., red or blue) represents the magnitude of the correlation coefficient:
Redder shades for positive correlations.
Bluer shades for negative correlations.
The intensity of the color shows the strength of the correlation.

# Detailed Example from our Image:
The correlation between YearsExperience and Salary is 0.97. This indicates a very strong positive linear relationship, meaning as the experience increases, the salary tends to increase as well.
The heatmap uses red for strong positive correlations and blue for weaker or negative correlations. Since all values are positive and close to 1, most cells are red.

# plt.plot(np.sort(X_train, axis=0), model_poly.predict(poly.transform(np.sort(X_train, axis=0))), color='red', label='Polynomial Curve')

Explanation:
# np.sort(X_train, axis=0):

Purpose: The X_train data contains the independent variable(s) (in this case, YearsExperience). This line sorts the training data (X_train) in ascending order.
Why do we sort it?: Sorting ensures that the polynomial regression curve is plotted smoothly. Without sorting, the curve might look jagged or disconnected because the points are not ordered by the x-axis values (Years of Experience). Sorting allows us to create a smooth line that follows the trend of the data.

# poly.transform(np.sort(X_train, axis=0)):

Purpose: poly is an instance of PolynomialFeatures(degree=3), which transforms the original feature (YearsExperience) into polynomial features. The transform() method applies the polynomial transformation to the sorted X_train data.
Why do we use it?: Polynomial regression works by fitting the model to transformed features (polynomial terms). In this case, for each value of YearsExperience, the model needs higher-degree features (like YearsExperience^2, YearsExperience^3, etc.) to fit a curve. So, we transform the sorted data into polynomial features before making predictions.

# model_poly.predict(poly.transform(np.sort(X_train, axis=0))):

Purpose: This applies the trained polynomial regression model (model_poly) to the transformed sorted X_train data and makes predictions for each sorted value of YearsExperience.
Why do we use it?: After transforming the sorted data into polynomial features, we want the model to predict the corresponding salary values for each of these sorted years of experience. The result will be a set of predicted salary values that we will plot as the polynomial regression curve.

# plt.plot(..., color='red', label='Polynomial Curve'):

Purpose: The plt.plot() function plots the line that represents the predicted salary values for the sorted X_train data (the polynomial regression curve).
Why do we use it?: The color='red' argument colors the polynomial curve red, and the label='Polynomial Curve' adds a label to this curve in the legend of the plot.
Why plot the curve?: This shows the predicted trend of the data based on the polynomial regression model, making it easier to visualize how well the model fits the data. The smoother, curved line represents the polynomial fit, while the scatter points show the actual data points.

In Simple Terms:
Sorting: You sort the YearsExperience values so that when you plot the curve, it will appear smooth rather than scattered.
Transforming: You convert the years of experience into a higher-degree polynomial form (e.g., YearsExperience^2, YearsExperience^3) to allow the curve to bend and fit the data more accurately.
Prediction: You then use the trained model to predict the salary values based on these transformed features.
Plotting: Finally, you plot a smooth red curve showing the model's predictions against the actual training data.