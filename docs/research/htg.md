# EDA: High Target Gradients

!!! tip inline end "SageWorks EDS"
    The SageWorks toolkit a set of plots that show EDA results, it also has a flexible plugin architecture to expand, enhance, or even replace the current set of web components [Dashboard](../aws_setup/dashboard_stack.md).

The SageWorks framework has a broad range of Exploratory Data Analysis (EDA) functionality. Each time a DataSource or FeatureSet is created that data is run through a full set of EDA techniques:

- TBD
- TBD2


One of the latest EDA techniques we've added is the addition of a concept called **High Target Gradients** 

1. **Definition**: For a given data point \(x_i\) with target value \(y_i\), and its neighbor \(x_j\) with target value \(y_j\), the target gradient \(G_{ij}\) can be defined as:

   \[G_{ij} = \frac{|y_i - y_j|}{d(x_i, x_j)}\]

   where \(d(x_i, x_j)\) is the distance between \(x_i\) and \(x_j\) in the feature space. This equation gives you the rate of change of the target value with respect to the change in features, similar to a slope in a two-dimensional space.

1. **Max Gradient for Each Point**: For each data point \(x_i\), you can compute the maximum target gradient with respect to all its neighbors:

   \[G_{i}^{max} = \max_{j \neq i} G_{ij}\]

   This gives you a scalar value for each point in your training data that represents the maximum rate of change of the target value in its local neighborhood.

1. **Usage**: You can use \(G_{i}^{max}\) to identify and filter areas in the feature space that have high target gradients, which may indicate potential issues with data quality or feature representation.

1. **Visualization**: Plotting the distribution of \(G_{i}^{max}\) values or visualizing them in the context of the feature space can help you identify regions or specific points that warrant further investigation.


## Additional Resources

<img align="right" src="../../images/scp.png" width="180">

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
