# Project Report: Dimensionality reduction

## Install Dependencies

```python
pip install numpy pandas
```

## Run

```bash
eg: python [program1] [input_data] [input_labels] [output_vector] [output_reduced_data]
```

## PART I

1. pca1.py: PCA without subtracting the mean

2. pca2.py : PCA with mean subtraction

3. scatter1.py : Minimizes the within-class scatter

4. scatter2.py : Maximizes the between-class scatter

5. scatter3.py : Maximize the ratio of between-class scatter and within-class scatter

## PART II

Programs are implemented for 2 cases:
In the first case all items in the training data are candidates for the nearest neighbor. (when label = -1) In the second case the nearest neighbor is a specific label. (when label = 1/2/3) Implementation

1. reducedim1.py (case 1)
2. reducedim2.py (case 2)
