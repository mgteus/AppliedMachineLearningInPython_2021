


---

# Assignment 1 - Introduction to Machine Learning

For this assignment, you will be using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients. First, read through the description of the dataset (below).


```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer) # Print the data set description
```

    {'data': array([[  1.79900000e+01,   1.03800000e+01,   1.22800000e+02, ...,
              2.65400000e-01,   4.60100000e-01,   1.18900000e-01],
           [  2.05700000e+01,   1.77700000e+01,   1.32900000e+02, ...,
              1.86000000e-01,   2.75000000e-01,   8.90200000e-02],
           [  1.96900000e+01,   2.12500000e+01,   1.30000000e+02, ...,
              2.43000000e-01,   3.61300000e-01,   8.75800000e-02],
           ..., 
           [  1.66000000e+01,   2.80800000e+01,   1.08300000e+02, ...,
              1.41800000e-01,   2.21800000e-01,   7.82000000e-02],
           [  2.06000000e+01,   2.93300000e+01,   1.40100000e+02, ...,
              2.65000000e-01,   4.08700000e-01,   1.24000000e-01],
           [  7.76000000e+00,   2.45400000e+01,   4.79200000e+01, ...,
              0.00000000e+00,   2.87100000e-01,   7.03900000e-02]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
           1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0,
           1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1,
           0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
           0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
           0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
           0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
           1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
           1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
           0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
           1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]), 'target_names': array(['malignant', 'benign'], 
          dtype='<U9'), 'DESCR': 'Breast Cancer Wisconsin (Diagnostic) Database\n=============================================\n\nNotes\n-----\nData Set Characteristics:\n    :Number of Instances: 569\n\n    :Number of Attributes: 30 numeric, predictive attributes and the class\n\n    :Attribute Information:\n        - radius (mean of distances from center to points on the perimeter)\n        - texture (standard deviation of gray-scale values)\n        - perimeter\n        - area\n        - smoothness (local variation in radius lengths)\n        - compactness (perimeter^2 / area - 1.0)\n        - concavity (severity of concave portions of the contour)\n        - concave points (number of concave portions of the contour)\n        - symmetry \n        - fractal dimension ("coastline approximation" - 1)\n\n        The mean, standard error, and "worst" or largest (mean of the three\n        largest values) of these features were computed for each image,\n        resulting in 30 features.  For instance, field 3 is Mean Radius, field\n        13 is Radius SE, field 23 is Worst Radius.\n\n        - class:\n                - WDBC-Malignant\n                - WDBC-Benign\n\n    :Summary Statistics:\n\n    ===================================== ====== ======\n                                           Min    Max\n    ===================================== ====== ======\n    radius (mean):                        6.981  28.11\n    texture (mean):                       9.71   39.28\n    perimeter (mean):                     43.79  188.5\n    area (mean):                          143.5  2501.0\n    smoothness (mean):                    0.053  0.163\n    compactness (mean):                   0.019  0.345\n    concavity (mean):                     0.0    0.427\n    concave points (mean):                0.0    0.201\n    symmetry (mean):                      0.106  0.304\n    fractal dimension (mean):             0.05   0.097\n    radius (standard error):              0.112  2.873\n    texture (standard error):             0.36   4.885\n    perimeter (standard error):           0.757  21.98\n    area (standard error):                6.802  542.2\n    smoothness (standard error):          0.002  0.031\n    compactness (standard error):         0.002  0.135\n    concavity (standard error):           0.0    0.396\n    concave points (standard error):      0.0    0.053\n    symmetry (standard error):            0.008  0.079\n    fractal dimension (standard error):   0.001  0.03\n    radius (worst):                       7.93   36.04\n    texture (worst):                      12.02  49.54\n    perimeter (worst):                    50.41  251.2\n    area (worst):                         185.2  4254.0\n    smoothness (worst):                   0.071  0.223\n    compactness (worst):                  0.027  1.058\n    concavity (worst):                    0.0    1.252\n    concave points (worst):               0.0    0.291\n    symmetry (worst):                     0.156  0.664\n    fractal dimension (worst):            0.055  0.208\n    ===================================== ====== ======\n\n    :Missing Attribute Values: None\n\n    :Class Distribution: 212 - Malignant, 357 - Benign\n\n    :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\n\n    :Donor: Nick Street\n\n    :Date: November, 1995\n\nThis is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\nhttps://goo.gl/U2Uwz2\n\nFeatures are computed from a digitized image of a fine needle\naspirate (FNA) of a breast mass.  They describe\ncharacteristics of the cell nuclei present in the image.\n\nSeparating plane described above was obtained using\nMultisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree\nConstruction Via Linear Programming." Proceedings of the 4th\nMidwest Artificial Intelligence and Cognitive Science Society,\npp. 97-101, 1992], a classification method which uses linear\nprogramming to construct a decision tree.  Relevant features\nwere selected using an exhaustive search in the space of 1-4\nfeatures and 1-3 separating planes.\n\nThe actual linear program used to obtain the separating plane\nin the 3-dimensional space is that described in:\n[K. P. Bennett and O. L. Mangasarian: "Robust Linear\nProgramming Discrimination of Two Linearly Inseparable Sets",\nOptimization Methods and Software 1, 1992, 23-34].\n\nThis database is also available through the UW CS ftp server:\n\nftp ftp.cs.wisc.edu\ncd math-prog/cpo-dataset/machine-learn/WDBC/\n\nReferences\n----------\n   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction \n     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on \n     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,\n     San Jose, CA, 1993.\n   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and \n     prognosis via linear programming. Operations Research, 43(4), pages 570-577, \n     July-August 1995.\n   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques\n     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) \n     163-171.\n', 'feature_names': array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error', 'fractal dimension error',
           'worst radius', 'worst texture', 'worst perimeter', 'worst area',
           'worst smoothness', 'worst compactness', 'worst concavity',
           'worst concave points', 'worst symmetry', 'worst fractal dimension'], 
          dtype='<U23')}


The object returned by `load_breast_cancer()` is a scikit-learn Bunch object, which is similar to a dictionary.


```python
cancer.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])



### Question 0 (Example)

How many features does the breast cancer dataset have?

*This function should return an integer.*


```python
# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 
```




    30



### Question 1

Scikit-learn works with lists, numpy arrays, scipy-sparse matrices, and pandas DataFrames, so converting the dataset to a DataFrame is not necessary for training this model. Using a DataFrame does however help make many things easier such as munging data, so let's practice creating a classifier with a pandas DataFrame. 



Convert the sklearn.dataset `cancer` to a DataFrame. 

*This function should return a `(569, 31)` DataFrame with * 

*columns = *

    ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension',
    'target']

*and index = *

    RangeIndex(start=0, stop=569, step=1)


```python



def answer_one():
    cols =['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension',
    'target']
    
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    
    return df
answer_one()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.990</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.300100</td>
      <td>0.147100</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.16220</td>
      <td>0.66560</td>
      <td>0.71190</td>
      <td>0.26540</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.570</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.086900</td>
      <td>0.070170</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.12380</td>
      <td>0.18660</td>
      <td>0.24160</td>
      <td>0.18600</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.690</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.197400</td>
      <td>0.127900</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.14440</td>
      <td>0.42450</td>
      <td>0.45040</td>
      <td>0.24300</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.420</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.241400</td>
      <td>0.105200</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.20980</td>
      <td>0.86630</td>
      <td>0.68690</td>
      <td>0.25750</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.290</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.198000</td>
      <td>0.104300</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.13740</td>
      <td>0.20500</td>
      <td>0.40000</td>
      <td>0.16250</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.450</td>
      <td>15.70</td>
      <td>82.57</td>
      <td>477.1</td>
      <td>0.12780</td>
      <td>0.17000</td>
      <td>0.157800</td>
      <td>0.080890</td>
      <td>0.2087</td>
      <td>0.07613</td>
      <td>...</td>
      <td>23.75</td>
      <td>103.40</td>
      <td>741.6</td>
      <td>0.17910</td>
      <td>0.52490</td>
      <td>0.53550</td>
      <td>0.17410</td>
      <td>0.3985</td>
      <td>0.12440</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18.250</td>
      <td>19.98</td>
      <td>119.60</td>
      <td>1040.0</td>
      <td>0.09463</td>
      <td>0.10900</td>
      <td>0.112700</td>
      <td>0.074000</td>
      <td>0.1794</td>
      <td>0.05742</td>
      <td>...</td>
      <td>27.66</td>
      <td>153.20</td>
      <td>1606.0</td>
      <td>0.14420</td>
      <td>0.25760</td>
      <td>0.37840</td>
      <td>0.19320</td>
      <td>0.3063</td>
      <td>0.08368</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13.710</td>
      <td>20.83</td>
      <td>90.20</td>
      <td>577.9</td>
      <td>0.11890</td>
      <td>0.16450</td>
      <td>0.093660</td>
      <td>0.059850</td>
      <td>0.2196</td>
      <td>0.07451</td>
      <td>...</td>
      <td>28.14</td>
      <td>110.60</td>
      <td>897.0</td>
      <td>0.16540</td>
      <td>0.36820</td>
      <td>0.26780</td>
      <td>0.15560</td>
      <td>0.3196</td>
      <td>0.11510</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.000</td>
      <td>21.82</td>
      <td>87.50</td>
      <td>519.8</td>
      <td>0.12730</td>
      <td>0.19320</td>
      <td>0.185900</td>
      <td>0.093530</td>
      <td>0.2350</td>
      <td>0.07389</td>
      <td>...</td>
      <td>30.73</td>
      <td>106.20</td>
      <td>739.3</td>
      <td>0.17030</td>
      <td>0.54010</td>
      <td>0.53900</td>
      <td>0.20600</td>
      <td>0.4378</td>
      <td>0.10720</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12.460</td>
      <td>24.04</td>
      <td>83.97</td>
      <td>475.9</td>
      <td>0.11860</td>
      <td>0.23960</td>
      <td>0.227300</td>
      <td>0.085430</td>
      <td>0.2030</td>
      <td>0.08243</td>
      <td>...</td>
      <td>40.68</td>
      <td>97.65</td>
      <td>711.4</td>
      <td>0.18530</td>
      <td>1.05800</td>
      <td>1.10500</td>
      <td>0.22100</td>
      <td>0.4366</td>
      <td>0.20750</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>16.020</td>
      <td>23.24</td>
      <td>102.70</td>
      <td>797.8</td>
      <td>0.08206</td>
      <td>0.06669</td>
      <td>0.032990</td>
      <td>0.033230</td>
      <td>0.1528</td>
      <td>0.05697</td>
      <td>...</td>
      <td>33.88</td>
      <td>123.80</td>
      <td>1150.0</td>
      <td>0.11810</td>
      <td>0.15510</td>
      <td>0.14590</td>
      <td>0.09975</td>
      <td>0.2948</td>
      <td>0.08452</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15.780</td>
      <td>17.89</td>
      <td>103.60</td>
      <td>781.0</td>
      <td>0.09710</td>
      <td>0.12920</td>
      <td>0.099540</td>
      <td>0.066060</td>
      <td>0.1842</td>
      <td>0.06082</td>
      <td>...</td>
      <td>27.28</td>
      <td>136.50</td>
      <td>1299.0</td>
      <td>0.13960</td>
      <td>0.56090</td>
      <td>0.39650</td>
      <td>0.18100</td>
      <td>0.3792</td>
      <td>0.10480</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>19.170</td>
      <td>24.80</td>
      <td>132.40</td>
      <td>1123.0</td>
      <td>0.09740</td>
      <td>0.24580</td>
      <td>0.206500</td>
      <td>0.111800</td>
      <td>0.2397</td>
      <td>0.07800</td>
      <td>...</td>
      <td>29.94</td>
      <td>151.70</td>
      <td>1332.0</td>
      <td>0.10370</td>
      <td>0.39030</td>
      <td>0.36390</td>
      <td>0.17670</td>
      <td>0.3176</td>
      <td>0.10230</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15.850</td>
      <td>23.95</td>
      <td>103.70</td>
      <td>782.7</td>
      <td>0.08401</td>
      <td>0.10020</td>
      <td>0.099380</td>
      <td>0.053640</td>
      <td>0.1847</td>
      <td>0.05338</td>
      <td>...</td>
      <td>27.66</td>
      <td>112.00</td>
      <td>876.5</td>
      <td>0.11310</td>
      <td>0.19240</td>
      <td>0.23220</td>
      <td>0.11190</td>
      <td>0.2809</td>
      <td>0.06287</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13.730</td>
      <td>22.61</td>
      <td>93.60</td>
      <td>578.3</td>
      <td>0.11310</td>
      <td>0.22930</td>
      <td>0.212800</td>
      <td>0.080250</td>
      <td>0.2069</td>
      <td>0.07682</td>
      <td>...</td>
      <td>32.01</td>
      <td>108.80</td>
      <td>697.7</td>
      <td>0.16510</td>
      <td>0.77250</td>
      <td>0.69430</td>
      <td>0.22080</td>
      <td>0.3596</td>
      <td>0.14310</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>14.540</td>
      <td>27.54</td>
      <td>96.73</td>
      <td>658.8</td>
      <td>0.11390</td>
      <td>0.15950</td>
      <td>0.163900</td>
      <td>0.073640</td>
      <td>0.2303</td>
      <td>0.07077</td>
      <td>...</td>
      <td>37.13</td>
      <td>124.10</td>
      <td>943.2</td>
      <td>0.16780</td>
      <td>0.65770</td>
      <td>0.70260</td>
      <td>0.17120</td>
      <td>0.4218</td>
      <td>0.13410</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>14.680</td>
      <td>20.13</td>
      <td>94.74</td>
      <td>684.5</td>
      <td>0.09867</td>
      <td>0.07200</td>
      <td>0.073950</td>
      <td>0.052590</td>
      <td>0.1586</td>
      <td>0.05922</td>
      <td>...</td>
      <td>30.88</td>
      <td>123.40</td>
      <td>1138.0</td>
      <td>0.14640</td>
      <td>0.18710</td>
      <td>0.29140</td>
      <td>0.16090</td>
      <td>0.3029</td>
      <td>0.08216</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>16.130</td>
      <td>20.68</td>
      <td>108.10</td>
      <td>798.8</td>
      <td>0.11700</td>
      <td>0.20220</td>
      <td>0.172200</td>
      <td>0.102800</td>
      <td>0.2164</td>
      <td>0.07356</td>
      <td>...</td>
      <td>31.48</td>
      <td>136.80</td>
      <td>1315.0</td>
      <td>0.17890</td>
      <td>0.42330</td>
      <td>0.47840</td>
      <td>0.20730</td>
      <td>0.3706</td>
      <td>0.11420</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19.810</td>
      <td>22.15</td>
      <td>130.00</td>
      <td>1260.0</td>
      <td>0.09831</td>
      <td>0.10270</td>
      <td>0.147900</td>
      <td>0.094980</td>
      <td>0.1582</td>
      <td>0.05395</td>
      <td>...</td>
      <td>30.88</td>
      <td>186.80</td>
      <td>2398.0</td>
      <td>0.15120</td>
      <td>0.31500</td>
      <td>0.53720</td>
      <td>0.23880</td>
      <td>0.2768</td>
      <td>0.07615</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>13.540</td>
      <td>14.36</td>
      <td>87.46</td>
      <td>566.3</td>
      <td>0.09779</td>
      <td>0.08129</td>
      <td>0.066640</td>
      <td>0.047810</td>
      <td>0.1885</td>
      <td>0.05766</td>
      <td>...</td>
      <td>19.26</td>
      <td>99.70</td>
      <td>711.2</td>
      <td>0.14400</td>
      <td>0.17730</td>
      <td>0.23900</td>
      <td>0.12880</td>
      <td>0.2977</td>
      <td>0.07259</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>13.080</td>
      <td>15.71</td>
      <td>85.63</td>
      <td>520.0</td>
      <td>0.10750</td>
      <td>0.12700</td>
      <td>0.045680</td>
      <td>0.031100</td>
      <td>0.1967</td>
      <td>0.06811</td>
      <td>...</td>
      <td>20.49</td>
      <td>96.09</td>
      <td>630.5</td>
      <td>0.13120</td>
      <td>0.27760</td>
      <td>0.18900</td>
      <td>0.07283</td>
      <td>0.3184</td>
      <td>0.08183</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9.504</td>
      <td>12.44</td>
      <td>60.34</td>
      <td>273.9</td>
      <td>0.10240</td>
      <td>0.06492</td>
      <td>0.029560</td>
      <td>0.020760</td>
      <td>0.1815</td>
      <td>0.06905</td>
      <td>...</td>
      <td>15.66</td>
      <td>65.13</td>
      <td>314.9</td>
      <td>0.13240</td>
      <td>0.11480</td>
      <td>0.08867</td>
      <td>0.06227</td>
      <td>0.2450</td>
      <td>0.07773</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>15.340</td>
      <td>14.26</td>
      <td>102.50</td>
      <td>704.4</td>
      <td>0.10730</td>
      <td>0.21350</td>
      <td>0.207700</td>
      <td>0.097560</td>
      <td>0.2521</td>
      <td>0.07032</td>
      <td>...</td>
      <td>19.08</td>
      <td>125.10</td>
      <td>980.9</td>
      <td>0.13900</td>
      <td>0.59540</td>
      <td>0.63050</td>
      <td>0.23930</td>
      <td>0.4667</td>
      <td>0.09946</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>21.160</td>
      <td>23.04</td>
      <td>137.20</td>
      <td>1404.0</td>
      <td>0.09428</td>
      <td>0.10220</td>
      <td>0.109700</td>
      <td>0.086320</td>
      <td>0.1769</td>
      <td>0.05278</td>
      <td>...</td>
      <td>35.59</td>
      <td>188.00</td>
      <td>2615.0</td>
      <td>0.14010</td>
      <td>0.26000</td>
      <td>0.31550</td>
      <td>0.20090</td>
      <td>0.2822</td>
      <td>0.07526</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>16.650</td>
      <td>21.38</td>
      <td>110.00</td>
      <td>904.6</td>
      <td>0.11210</td>
      <td>0.14570</td>
      <td>0.152500</td>
      <td>0.091700</td>
      <td>0.1995</td>
      <td>0.06330</td>
      <td>...</td>
      <td>31.56</td>
      <td>177.00</td>
      <td>2215.0</td>
      <td>0.18050</td>
      <td>0.35780</td>
      <td>0.46950</td>
      <td>0.20950</td>
      <td>0.3613</td>
      <td>0.09564</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>17.140</td>
      <td>16.40</td>
      <td>116.00</td>
      <td>912.7</td>
      <td>0.11860</td>
      <td>0.22760</td>
      <td>0.222900</td>
      <td>0.140100</td>
      <td>0.3040</td>
      <td>0.07413</td>
      <td>...</td>
      <td>21.40</td>
      <td>152.40</td>
      <td>1461.0</td>
      <td>0.15450</td>
      <td>0.39490</td>
      <td>0.38530</td>
      <td>0.25500</td>
      <td>0.4066</td>
      <td>0.10590</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>14.580</td>
      <td>21.53</td>
      <td>97.41</td>
      <td>644.8</td>
      <td>0.10540</td>
      <td>0.18680</td>
      <td>0.142500</td>
      <td>0.087830</td>
      <td>0.2252</td>
      <td>0.06924</td>
      <td>...</td>
      <td>33.21</td>
      <td>122.40</td>
      <td>896.9</td>
      <td>0.15250</td>
      <td>0.66430</td>
      <td>0.55390</td>
      <td>0.27010</td>
      <td>0.4264</td>
      <td>0.12750</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>18.610</td>
      <td>20.25</td>
      <td>122.10</td>
      <td>1094.0</td>
      <td>0.09440</td>
      <td>0.10660</td>
      <td>0.149000</td>
      <td>0.077310</td>
      <td>0.1697</td>
      <td>0.05699</td>
      <td>...</td>
      <td>27.26</td>
      <td>139.90</td>
      <td>1403.0</td>
      <td>0.13380</td>
      <td>0.21170</td>
      <td>0.34460</td>
      <td>0.14900</td>
      <td>0.2341</td>
      <td>0.07421</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>15.300</td>
      <td>25.27</td>
      <td>102.40</td>
      <td>732.4</td>
      <td>0.10820</td>
      <td>0.16970</td>
      <td>0.168300</td>
      <td>0.087510</td>
      <td>0.1926</td>
      <td>0.06540</td>
      <td>...</td>
      <td>36.71</td>
      <td>149.30</td>
      <td>1269.0</td>
      <td>0.16410</td>
      <td>0.61100</td>
      <td>0.63350</td>
      <td>0.20240</td>
      <td>0.4027</td>
      <td>0.09876</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>17.570</td>
      <td>15.05</td>
      <td>115.00</td>
      <td>955.1</td>
      <td>0.09847</td>
      <td>0.11570</td>
      <td>0.098750</td>
      <td>0.079530</td>
      <td>0.1739</td>
      <td>0.06149</td>
      <td>...</td>
      <td>19.52</td>
      <td>134.90</td>
      <td>1227.0</td>
      <td>0.12550</td>
      <td>0.28120</td>
      <td>0.24890</td>
      <td>0.14560</td>
      <td>0.2756</td>
      <td>0.07919</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>539</th>
      <td>7.691</td>
      <td>25.44</td>
      <td>48.34</td>
      <td>170.4</td>
      <td>0.08668</td>
      <td>0.11990</td>
      <td>0.092520</td>
      <td>0.013640</td>
      <td>0.2037</td>
      <td>0.07751</td>
      <td>...</td>
      <td>31.89</td>
      <td>54.49</td>
      <td>223.6</td>
      <td>0.15960</td>
      <td>0.30640</td>
      <td>0.33930</td>
      <td>0.05000</td>
      <td>0.2790</td>
      <td>0.10660</td>
      <td>1</td>
    </tr>
    <tr>
      <th>540</th>
      <td>11.540</td>
      <td>14.44</td>
      <td>74.65</td>
      <td>402.9</td>
      <td>0.09984</td>
      <td>0.11200</td>
      <td>0.067370</td>
      <td>0.025940</td>
      <td>0.1818</td>
      <td>0.06782</td>
      <td>...</td>
      <td>19.68</td>
      <td>78.78</td>
      <td>457.8</td>
      <td>0.13450</td>
      <td>0.21180</td>
      <td>0.17970</td>
      <td>0.06918</td>
      <td>0.2329</td>
      <td>0.08134</td>
      <td>1</td>
    </tr>
    <tr>
      <th>541</th>
      <td>14.470</td>
      <td>24.99</td>
      <td>95.81</td>
      <td>656.4</td>
      <td>0.08837</td>
      <td>0.12300</td>
      <td>0.100900</td>
      <td>0.038900</td>
      <td>0.1872</td>
      <td>0.06341</td>
      <td>...</td>
      <td>31.73</td>
      <td>113.50</td>
      <td>808.9</td>
      <td>0.13400</td>
      <td>0.42020</td>
      <td>0.40400</td>
      <td>0.12050</td>
      <td>0.3187</td>
      <td>0.10230</td>
      <td>1</td>
    </tr>
    <tr>
      <th>542</th>
      <td>14.740</td>
      <td>25.42</td>
      <td>94.70</td>
      <td>668.6</td>
      <td>0.08275</td>
      <td>0.07214</td>
      <td>0.041050</td>
      <td>0.030270</td>
      <td>0.1840</td>
      <td>0.05680</td>
      <td>...</td>
      <td>32.29</td>
      <td>107.40</td>
      <td>826.4</td>
      <td>0.10600</td>
      <td>0.13760</td>
      <td>0.16110</td>
      <td>0.10950</td>
      <td>0.2722</td>
      <td>0.06956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>543</th>
      <td>13.210</td>
      <td>28.06</td>
      <td>84.88</td>
      <td>538.4</td>
      <td>0.08671</td>
      <td>0.06877</td>
      <td>0.029870</td>
      <td>0.032750</td>
      <td>0.1628</td>
      <td>0.05781</td>
      <td>...</td>
      <td>37.17</td>
      <td>92.48</td>
      <td>629.6</td>
      <td>0.10720</td>
      <td>0.13810</td>
      <td>0.10620</td>
      <td>0.07958</td>
      <td>0.2473</td>
      <td>0.06443</td>
      <td>1</td>
    </tr>
    <tr>
      <th>544</th>
      <td>13.870</td>
      <td>20.70</td>
      <td>89.77</td>
      <td>584.8</td>
      <td>0.09578</td>
      <td>0.10180</td>
      <td>0.036880</td>
      <td>0.023690</td>
      <td>0.1620</td>
      <td>0.06688</td>
      <td>...</td>
      <td>24.75</td>
      <td>99.17</td>
      <td>688.6</td>
      <td>0.12640</td>
      <td>0.20370</td>
      <td>0.13770</td>
      <td>0.06845</td>
      <td>0.2249</td>
      <td>0.08492</td>
      <td>1</td>
    </tr>
    <tr>
      <th>545</th>
      <td>13.620</td>
      <td>23.23</td>
      <td>87.19</td>
      <td>573.2</td>
      <td>0.09246</td>
      <td>0.06747</td>
      <td>0.029740</td>
      <td>0.024430</td>
      <td>0.1664</td>
      <td>0.05801</td>
      <td>...</td>
      <td>29.09</td>
      <td>97.58</td>
      <td>729.8</td>
      <td>0.12160</td>
      <td>0.15170</td>
      <td>0.10490</td>
      <td>0.07174</td>
      <td>0.2642</td>
      <td>0.06953</td>
      <td>1</td>
    </tr>
    <tr>
      <th>546</th>
      <td>10.320</td>
      <td>16.35</td>
      <td>65.31</td>
      <td>324.9</td>
      <td>0.09434</td>
      <td>0.04994</td>
      <td>0.010120</td>
      <td>0.005495</td>
      <td>0.1885</td>
      <td>0.06201</td>
      <td>...</td>
      <td>21.77</td>
      <td>71.12</td>
      <td>384.9</td>
      <td>0.12850</td>
      <td>0.08842</td>
      <td>0.04384</td>
      <td>0.02381</td>
      <td>0.2681</td>
      <td>0.07399</td>
      <td>1</td>
    </tr>
    <tr>
      <th>547</th>
      <td>10.260</td>
      <td>16.58</td>
      <td>65.85</td>
      <td>320.8</td>
      <td>0.08877</td>
      <td>0.08066</td>
      <td>0.043580</td>
      <td>0.024380</td>
      <td>0.1669</td>
      <td>0.06714</td>
      <td>...</td>
      <td>22.04</td>
      <td>71.08</td>
      <td>357.4</td>
      <td>0.14610</td>
      <td>0.22460</td>
      <td>0.17830</td>
      <td>0.08333</td>
      <td>0.2691</td>
      <td>0.09479</td>
      <td>1</td>
    </tr>
    <tr>
      <th>548</th>
      <td>9.683</td>
      <td>19.34</td>
      <td>61.05</td>
      <td>285.7</td>
      <td>0.08491</td>
      <td>0.05030</td>
      <td>0.023370</td>
      <td>0.009615</td>
      <td>0.1580</td>
      <td>0.06235</td>
      <td>...</td>
      <td>25.59</td>
      <td>69.10</td>
      <td>364.2</td>
      <td>0.11990</td>
      <td>0.09546</td>
      <td>0.09350</td>
      <td>0.03846</td>
      <td>0.2552</td>
      <td>0.07920</td>
      <td>1</td>
    </tr>
    <tr>
      <th>549</th>
      <td>10.820</td>
      <td>24.21</td>
      <td>68.89</td>
      <td>361.6</td>
      <td>0.08192</td>
      <td>0.06602</td>
      <td>0.015480</td>
      <td>0.008160</td>
      <td>0.1976</td>
      <td>0.06328</td>
      <td>...</td>
      <td>31.45</td>
      <td>83.90</td>
      <td>505.6</td>
      <td>0.12040</td>
      <td>0.16330</td>
      <td>0.06194</td>
      <td>0.03264</td>
      <td>0.3059</td>
      <td>0.07626</td>
      <td>1</td>
    </tr>
    <tr>
      <th>550</th>
      <td>10.860</td>
      <td>21.48</td>
      <td>68.51</td>
      <td>360.5</td>
      <td>0.07431</td>
      <td>0.04227</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1661</td>
      <td>0.05948</td>
      <td>...</td>
      <td>24.77</td>
      <td>74.08</td>
      <td>412.3</td>
      <td>0.10010</td>
      <td>0.07348</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.2458</td>
      <td>0.06592</td>
      <td>1</td>
    </tr>
    <tr>
      <th>551</th>
      <td>11.130</td>
      <td>22.44</td>
      <td>71.49</td>
      <td>378.4</td>
      <td>0.09566</td>
      <td>0.08194</td>
      <td>0.048240</td>
      <td>0.022570</td>
      <td>0.2030</td>
      <td>0.06552</td>
      <td>...</td>
      <td>28.26</td>
      <td>77.80</td>
      <td>436.6</td>
      <td>0.10870</td>
      <td>0.17820</td>
      <td>0.15640</td>
      <td>0.06413</td>
      <td>0.3169</td>
      <td>0.08032</td>
      <td>1</td>
    </tr>
    <tr>
      <th>552</th>
      <td>12.770</td>
      <td>29.43</td>
      <td>81.35</td>
      <td>507.9</td>
      <td>0.08276</td>
      <td>0.04234</td>
      <td>0.019970</td>
      <td>0.014990</td>
      <td>0.1539</td>
      <td>0.05637</td>
      <td>...</td>
      <td>36.00</td>
      <td>88.10</td>
      <td>594.7</td>
      <td>0.12340</td>
      <td>0.10640</td>
      <td>0.08653</td>
      <td>0.06498</td>
      <td>0.2407</td>
      <td>0.06484</td>
      <td>1</td>
    </tr>
    <tr>
      <th>553</th>
      <td>9.333</td>
      <td>21.94</td>
      <td>59.01</td>
      <td>264.0</td>
      <td>0.09240</td>
      <td>0.05605</td>
      <td>0.039960</td>
      <td>0.012820</td>
      <td>0.1692</td>
      <td>0.06576</td>
      <td>...</td>
      <td>25.05</td>
      <td>62.86</td>
      <td>295.8</td>
      <td>0.11030</td>
      <td>0.08298</td>
      <td>0.07993</td>
      <td>0.02564</td>
      <td>0.2435</td>
      <td>0.07393</td>
      <td>1</td>
    </tr>
    <tr>
      <th>554</th>
      <td>12.880</td>
      <td>28.92</td>
      <td>82.50</td>
      <td>514.3</td>
      <td>0.08123</td>
      <td>0.05824</td>
      <td>0.061950</td>
      <td>0.023430</td>
      <td>0.1566</td>
      <td>0.05708</td>
      <td>...</td>
      <td>35.74</td>
      <td>88.84</td>
      <td>595.7</td>
      <td>0.12270</td>
      <td>0.16200</td>
      <td>0.24390</td>
      <td>0.06493</td>
      <td>0.2372</td>
      <td>0.07242</td>
      <td>1</td>
    </tr>
    <tr>
      <th>555</th>
      <td>10.290</td>
      <td>27.61</td>
      <td>65.67</td>
      <td>321.4</td>
      <td>0.09030</td>
      <td>0.07658</td>
      <td>0.059990</td>
      <td>0.027380</td>
      <td>0.1593</td>
      <td>0.06127</td>
      <td>...</td>
      <td>34.91</td>
      <td>69.57</td>
      <td>357.6</td>
      <td>0.13840</td>
      <td>0.17100</td>
      <td>0.20000</td>
      <td>0.09127</td>
      <td>0.2226</td>
      <td>0.08283</td>
      <td>1</td>
    </tr>
    <tr>
      <th>556</th>
      <td>10.160</td>
      <td>19.59</td>
      <td>64.73</td>
      <td>311.7</td>
      <td>0.10030</td>
      <td>0.07504</td>
      <td>0.005025</td>
      <td>0.011160</td>
      <td>0.1791</td>
      <td>0.06331</td>
      <td>...</td>
      <td>22.88</td>
      <td>67.88</td>
      <td>347.3</td>
      <td>0.12650</td>
      <td>0.12000</td>
      <td>0.01005</td>
      <td>0.02232</td>
      <td>0.2262</td>
      <td>0.06742</td>
      <td>1</td>
    </tr>
    <tr>
      <th>557</th>
      <td>9.423</td>
      <td>27.88</td>
      <td>59.26</td>
      <td>271.3</td>
      <td>0.08123</td>
      <td>0.04971</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1742</td>
      <td>0.06059</td>
      <td>...</td>
      <td>34.24</td>
      <td>66.50</td>
      <td>330.6</td>
      <td>0.10730</td>
      <td>0.07158</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.2475</td>
      <td>0.06969</td>
      <td>1</td>
    </tr>
    <tr>
      <th>558</th>
      <td>14.590</td>
      <td>22.68</td>
      <td>96.39</td>
      <td>657.1</td>
      <td>0.08473</td>
      <td>0.13300</td>
      <td>0.102900</td>
      <td>0.037360</td>
      <td>0.1454</td>
      <td>0.06147</td>
      <td>...</td>
      <td>27.27</td>
      <td>105.90</td>
      <td>733.5</td>
      <td>0.10260</td>
      <td>0.31710</td>
      <td>0.36620</td>
      <td>0.11050</td>
      <td>0.2258</td>
      <td>0.08004</td>
      <td>1</td>
    </tr>
    <tr>
      <th>559</th>
      <td>11.510</td>
      <td>23.93</td>
      <td>74.52</td>
      <td>403.5</td>
      <td>0.09261</td>
      <td>0.10210</td>
      <td>0.111200</td>
      <td>0.041050</td>
      <td>0.1388</td>
      <td>0.06570</td>
      <td>...</td>
      <td>37.16</td>
      <td>82.28</td>
      <td>474.2</td>
      <td>0.12980</td>
      <td>0.25170</td>
      <td>0.36300</td>
      <td>0.09653</td>
      <td>0.2112</td>
      <td>0.08732</td>
      <td>1</td>
    </tr>
    <tr>
      <th>560</th>
      <td>14.050</td>
      <td>27.15</td>
      <td>91.38</td>
      <td>600.4</td>
      <td>0.09929</td>
      <td>0.11260</td>
      <td>0.044620</td>
      <td>0.043040</td>
      <td>0.1537</td>
      <td>0.06171</td>
      <td>...</td>
      <td>33.17</td>
      <td>100.20</td>
      <td>706.7</td>
      <td>0.12410</td>
      <td>0.22640</td>
      <td>0.13260</td>
      <td>0.10480</td>
      <td>0.2250</td>
      <td>0.08321</td>
      <td>1</td>
    </tr>
    <tr>
      <th>561</th>
      <td>11.200</td>
      <td>29.37</td>
      <td>70.67</td>
      <td>386.0</td>
      <td>0.07449</td>
      <td>0.03558</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1060</td>
      <td>0.05502</td>
      <td>...</td>
      <td>38.30</td>
      <td>75.19</td>
      <td>439.6</td>
      <td>0.09267</td>
      <td>0.05494</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.1566</td>
      <td>0.05905</td>
      <td>1</td>
    </tr>
    <tr>
      <th>562</th>
      <td>15.220</td>
      <td>30.62</td>
      <td>103.40</td>
      <td>716.9</td>
      <td>0.10480</td>
      <td>0.20870</td>
      <td>0.255000</td>
      <td>0.094290</td>
      <td>0.2128</td>
      <td>0.07152</td>
      <td>...</td>
      <td>42.79</td>
      <td>128.70</td>
      <td>915.0</td>
      <td>0.14170</td>
      <td>0.79170</td>
      <td>1.17000</td>
      <td>0.23560</td>
      <td>0.4089</td>
      <td>0.14090</td>
      <td>0</td>
    </tr>
    <tr>
      <th>563</th>
      <td>20.920</td>
      <td>25.09</td>
      <td>143.00</td>
      <td>1347.0</td>
      <td>0.10990</td>
      <td>0.22360</td>
      <td>0.317400</td>
      <td>0.147400</td>
      <td>0.2149</td>
      <td>0.06879</td>
      <td>...</td>
      <td>29.41</td>
      <td>179.10</td>
      <td>1819.0</td>
      <td>0.14070</td>
      <td>0.41860</td>
      <td>0.65990</td>
      <td>0.25420</td>
      <td>0.2929</td>
      <td>0.09873</td>
      <td>0</td>
    </tr>
    <tr>
      <th>564</th>
      <td>21.560</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.243900</td>
      <td>0.138900</td>
      <td>0.1726</td>
      <td>0.05623</td>
      <td>...</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.41070</td>
      <td>0.22160</td>
      <td>0.2060</td>
      <td>0.07115</td>
      <td>0</td>
    </tr>
    <tr>
      <th>565</th>
      <td>20.130</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.144000</td>
      <td>0.097910</td>
      <td>0.1752</td>
      <td>0.05533</td>
      <td>...</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.32150</td>
      <td>0.16280</td>
      <td>0.2572</td>
      <td>0.06637</td>
      <td>0</td>
    </tr>
    <tr>
      <th>566</th>
      <td>16.600</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.092510</td>
      <td>0.053020</td>
      <td>0.1590</td>
      <td>0.05648</td>
      <td>...</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.34030</td>
      <td>0.14180</td>
      <td>0.2218</td>
      <td>0.07820</td>
      <td>0</td>
    </tr>
    <tr>
      <th>567</th>
      <td>20.600</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.351400</td>
      <td>0.152000</td>
      <td>0.2397</td>
      <td>0.07016</td>
      <td>...</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.93870</td>
      <td>0.26500</td>
      <td>0.4087</td>
      <td>0.12400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>568</th>
      <td>7.760</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1587</td>
      <td>0.05884</td>
      <td>...</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.2871</td>
      <td>0.07039</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 31 columns</p>
</div>



### Question 2
What is the class distribution? (i.e. how many instances of `malignant` (encoded 0) and how many `benign` (encoded 1)?)

*This function should return a Series named `target` of length 2 with integer values and index =* `['malignant', 'benign']`


```python
def answer_two():
    cancerdf = answer_one()
    #---- Masking target columns values equals to zero --------
    mask = cancerdf['target'] == 0
    #--- using '.shape[0]' to get the length -------------------
    q2_data = [cancerdf[mask].shape[0], cancerdf[~mask].shape[0]]
    
    #--- creating the pandas series ---------------------------
    target = pd.Series(q2_data, index=["malignant", "benign"])
    
    return target



answer_two()
```




    malignant    212
    benign       357
    dtype: int64



### Question 3
Split the DataFrame into `X` (the data) and `y` (the labels).

*This function should return a tuple of length 2:* `(X, y)`*, where* 
* `X`*, a pandas DataFrame, has shape* `(569, 30)`
* `y`*, a pandas Series, has shape* `(569,)`.


```python
def answer_three():
    cancerdf = answer_one()
    
    X = cancerdf.drop('target', axis=1).astype('float')
    y = cancerdf['target'].astype('float')
    
    return X, y



answer_three()
```




    (     mean radius  mean texture  mean perimeter  mean area  mean smoothness  \
     0         17.990         10.38          122.80     1001.0          0.11840   
     1         20.570         17.77          132.90     1326.0          0.08474   
     2         19.690         21.25          130.00     1203.0          0.10960   
     3         11.420         20.38           77.58      386.1          0.14250   
     4         20.290         14.34          135.10     1297.0          0.10030   
     5         12.450         15.70           82.57      477.1          0.12780   
     6         18.250         19.98          119.60     1040.0          0.09463   
     7         13.710         20.83           90.20      577.9          0.11890   
     8         13.000         21.82           87.50      519.8          0.12730   
     9         12.460         24.04           83.97      475.9          0.11860   
     10        16.020         23.24          102.70      797.8          0.08206   
     11        15.780         17.89          103.60      781.0          0.09710   
     12        19.170         24.80          132.40     1123.0          0.09740   
     13        15.850         23.95          103.70      782.7          0.08401   
     14        13.730         22.61           93.60      578.3          0.11310   
     15        14.540         27.54           96.73      658.8          0.11390   
     16        14.680         20.13           94.74      684.5          0.09867   
     17        16.130         20.68          108.10      798.8          0.11700   
     18        19.810         22.15          130.00     1260.0          0.09831   
     19        13.540         14.36           87.46      566.3          0.09779   
     20        13.080         15.71           85.63      520.0          0.10750   
     21         9.504         12.44           60.34      273.9          0.10240   
     22        15.340         14.26          102.50      704.4          0.10730   
     23        21.160         23.04          137.20     1404.0          0.09428   
     24        16.650         21.38          110.00      904.6          0.11210   
     25        17.140         16.40          116.00      912.7          0.11860   
     26        14.580         21.53           97.41      644.8          0.10540   
     27        18.610         20.25          122.10     1094.0          0.09440   
     28        15.300         25.27          102.40      732.4          0.10820   
     29        17.570         15.05          115.00      955.1          0.09847   
     ..           ...           ...             ...        ...              ...   
     539        7.691         25.44           48.34      170.4          0.08668   
     540       11.540         14.44           74.65      402.9          0.09984   
     541       14.470         24.99           95.81      656.4          0.08837   
     542       14.740         25.42           94.70      668.6          0.08275   
     543       13.210         28.06           84.88      538.4          0.08671   
     544       13.870         20.70           89.77      584.8          0.09578   
     545       13.620         23.23           87.19      573.2          0.09246   
     546       10.320         16.35           65.31      324.9          0.09434   
     547       10.260         16.58           65.85      320.8          0.08877   
     548        9.683         19.34           61.05      285.7          0.08491   
     549       10.820         24.21           68.89      361.6          0.08192   
     550       10.860         21.48           68.51      360.5          0.07431   
     551       11.130         22.44           71.49      378.4          0.09566   
     552       12.770         29.43           81.35      507.9          0.08276   
     553        9.333         21.94           59.01      264.0          0.09240   
     554       12.880         28.92           82.50      514.3          0.08123   
     555       10.290         27.61           65.67      321.4          0.09030   
     556       10.160         19.59           64.73      311.7          0.10030   
     557        9.423         27.88           59.26      271.3          0.08123   
     558       14.590         22.68           96.39      657.1          0.08473   
     559       11.510         23.93           74.52      403.5          0.09261   
     560       14.050         27.15           91.38      600.4          0.09929   
     561       11.200         29.37           70.67      386.0          0.07449   
     562       15.220         30.62          103.40      716.9          0.10480   
     563       20.920         25.09          143.00     1347.0          0.10990   
     564       21.560         22.39          142.00     1479.0          0.11100   
     565       20.130         28.25          131.20     1261.0          0.09780   
     566       16.600         28.08          108.30      858.1          0.08455   
     567       20.600         29.33          140.10     1265.0          0.11780   
     568        7.760         24.54           47.92      181.0          0.05263   
     
          mean compactness  mean concavity  mean concave points  mean symmetry  \
     0             0.27760        0.300100             0.147100         0.2419   
     1             0.07864        0.086900             0.070170         0.1812   
     2             0.15990        0.197400             0.127900         0.2069   
     3             0.28390        0.241400             0.105200         0.2597   
     4             0.13280        0.198000             0.104300         0.1809   
     5             0.17000        0.157800             0.080890         0.2087   
     6             0.10900        0.112700             0.074000         0.1794   
     7             0.16450        0.093660             0.059850         0.2196   
     8             0.19320        0.185900             0.093530         0.2350   
     9             0.23960        0.227300             0.085430         0.2030   
     10            0.06669        0.032990             0.033230         0.1528   
     11            0.12920        0.099540             0.066060         0.1842   
     12            0.24580        0.206500             0.111800         0.2397   
     13            0.10020        0.099380             0.053640         0.1847   
     14            0.22930        0.212800             0.080250         0.2069   
     15            0.15950        0.163900             0.073640         0.2303   
     16            0.07200        0.073950             0.052590         0.1586   
     17            0.20220        0.172200             0.102800         0.2164   
     18            0.10270        0.147900             0.094980         0.1582   
     19            0.08129        0.066640             0.047810         0.1885   
     20            0.12700        0.045680             0.031100         0.1967   
     21            0.06492        0.029560             0.020760         0.1815   
     22            0.21350        0.207700             0.097560         0.2521   
     23            0.10220        0.109700             0.086320         0.1769   
     24            0.14570        0.152500             0.091700         0.1995   
     25            0.22760        0.222900             0.140100         0.3040   
     26            0.18680        0.142500             0.087830         0.2252   
     27            0.10660        0.149000             0.077310         0.1697   
     28            0.16970        0.168300             0.087510         0.1926   
     29            0.11570        0.098750             0.079530         0.1739   
     ..                ...             ...                  ...            ...   
     539           0.11990        0.092520             0.013640         0.2037   
     540           0.11200        0.067370             0.025940         0.1818   
     541           0.12300        0.100900             0.038900         0.1872   
     542           0.07214        0.041050             0.030270         0.1840   
     543           0.06877        0.029870             0.032750         0.1628   
     544           0.10180        0.036880             0.023690         0.1620   
     545           0.06747        0.029740             0.024430         0.1664   
     546           0.04994        0.010120             0.005495         0.1885   
     547           0.08066        0.043580             0.024380         0.1669   
     548           0.05030        0.023370             0.009615         0.1580   
     549           0.06602        0.015480             0.008160         0.1976   
     550           0.04227        0.000000             0.000000         0.1661   
     551           0.08194        0.048240             0.022570         0.2030   
     552           0.04234        0.019970             0.014990         0.1539   
     553           0.05605        0.039960             0.012820         0.1692   
     554           0.05824        0.061950             0.023430         0.1566   
     555           0.07658        0.059990             0.027380         0.1593   
     556           0.07504        0.005025             0.011160         0.1791   
     557           0.04971        0.000000             0.000000         0.1742   
     558           0.13300        0.102900             0.037360         0.1454   
     559           0.10210        0.111200             0.041050         0.1388   
     560           0.11260        0.044620             0.043040         0.1537   
     561           0.03558        0.000000             0.000000         0.1060   
     562           0.20870        0.255000             0.094290         0.2128   
     563           0.22360        0.317400             0.147400         0.2149   
     564           0.11590        0.243900             0.138900         0.1726   
     565           0.10340        0.144000             0.097910         0.1752   
     566           0.10230        0.092510             0.053020         0.1590   
     567           0.27700        0.351400             0.152000         0.2397   
     568           0.04362        0.000000             0.000000         0.1587   
     
          mean fractal dimension           ...             worst radius  \
     0                   0.07871           ...                   25.380   
     1                   0.05667           ...                   24.990   
     2                   0.05999           ...                   23.570   
     3                   0.09744           ...                   14.910   
     4                   0.05883           ...                   22.540   
     5                   0.07613           ...                   15.470   
     6                   0.05742           ...                   22.880   
     7                   0.07451           ...                   17.060   
     8                   0.07389           ...                   15.490   
     9                   0.08243           ...                   15.090   
     10                  0.05697           ...                   19.190   
     11                  0.06082           ...                   20.420   
     12                  0.07800           ...                   20.960   
     13                  0.05338           ...                   16.840   
     14                  0.07682           ...                   15.030   
     15                  0.07077           ...                   17.460   
     16                  0.05922           ...                   19.070   
     17                  0.07356           ...                   20.960   
     18                  0.05395           ...                   27.320   
     19                  0.05766           ...                   15.110   
     20                  0.06811           ...                   14.500   
     21                  0.06905           ...                   10.230   
     22                  0.07032           ...                   18.070   
     23                  0.05278           ...                   29.170   
     24                  0.06330           ...                   26.460   
     25                  0.07413           ...                   22.250   
     26                  0.06924           ...                   17.620   
     27                  0.05699           ...                   21.310   
     28                  0.06540           ...                   20.270   
     29                  0.06149           ...                   20.010   
     ..                      ...           ...                      ...   
     539                 0.07751           ...                    8.678   
     540                 0.06782           ...                   12.260   
     541                 0.06341           ...                   16.220   
     542                 0.05680           ...                   16.510   
     543                 0.05781           ...                   14.370   
     544                 0.06688           ...                   15.050   
     545                 0.05801           ...                   15.350   
     546                 0.06201           ...                   11.250   
     547                 0.06714           ...                   10.830   
     548                 0.06235           ...                   10.930   
     549                 0.06328           ...                   13.030   
     550                 0.05948           ...                   11.660   
     551                 0.06552           ...                   12.020   
     552                 0.05637           ...                   13.870   
     553                 0.06576           ...                    9.845   
     554                 0.05708           ...                   13.890   
     555                 0.06127           ...                   10.840   
     556                 0.06331           ...                   10.650   
     557                 0.06059           ...                   10.490   
     558                 0.06147           ...                   15.480   
     559                 0.06570           ...                   12.480   
     560                 0.06171           ...                   15.300   
     561                 0.05502           ...                   11.920   
     562                 0.07152           ...                   17.520   
     563                 0.06879           ...                   24.290   
     564                 0.05623           ...                   25.450   
     565                 0.05533           ...                   23.690   
     566                 0.05648           ...                   18.980   
     567                 0.07016           ...                   25.740   
     568                 0.05884           ...                    9.456   
     
          worst texture  worst perimeter  worst area  worst smoothness  \
     0            17.33           184.60      2019.0           0.16220   
     1            23.41           158.80      1956.0           0.12380   
     2            25.53           152.50      1709.0           0.14440   
     3            26.50            98.87       567.7           0.20980   
     4            16.67           152.20      1575.0           0.13740   
     5            23.75           103.40       741.6           0.17910   
     6            27.66           153.20      1606.0           0.14420   
     7            28.14           110.60       897.0           0.16540   
     8            30.73           106.20       739.3           0.17030   
     9            40.68            97.65       711.4           0.18530   
     10           33.88           123.80      1150.0           0.11810   
     11           27.28           136.50      1299.0           0.13960   
     12           29.94           151.70      1332.0           0.10370   
     13           27.66           112.00       876.5           0.11310   
     14           32.01           108.80       697.7           0.16510   
     15           37.13           124.10       943.2           0.16780   
     16           30.88           123.40      1138.0           0.14640   
     17           31.48           136.80      1315.0           0.17890   
     18           30.88           186.80      2398.0           0.15120   
     19           19.26            99.70       711.2           0.14400   
     20           20.49            96.09       630.5           0.13120   
     21           15.66            65.13       314.9           0.13240   
     22           19.08           125.10       980.9           0.13900   
     23           35.59           188.00      2615.0           0.14010   
     24           31.56           177.00      2215.0           0.18050   
     25           21.40           152.40      1461.0           0.15450   
     26           33.21           122.40       896.9           0.15250   
     27           27.26           139.90      1403.0           0.13380   
     28           36.71           149.30      1269.0           0.16410   
     29           19.52           134.90      1227.0           0.12550   
     ..             ...              ...         ...               ...   
     539          31.89            54.49       223.6           0.15960   
     540          19.68            78.78       457.8           0.13450   
     541          31.73           113.50       808.9           0.13400   
     542          32.29           107.40       826.4           0.10600   
     543          37.17            92.48       629.6           0.10720   
     544          24.75            99.17       688.6           0.12640   
     545          29.09            97.58       729.8           0.12160   
     546          21.77            71.12       384.9           0.12850   
     547          22.04            71.08       357.4           0.14610   
     548          25.59            69.10       364.2           0.11990   
     549          31.45            83.90       505.6           0.12040   
     550          24.77            74.08       412.3           0.10010   
     551          28.26            77.80       436.6           0.10870   
     552          36.00            88.10       594.7           0.12340   
     553          25.05            62.86       295.8           0.11030   
     554          35.74            88.84       595.7           0.12270   
     555          34.91            69.57       357.6           0.13840   
     556          22.88            67.88       347.3           0.12650   
     557          34.24            66.50       330.6           0.10730   
     558          27.27           105.90       733.5           0.10260   
     559          37.16            82.28       474.2           0.12980   
     560          33.17           100.20       706.7           0.12410   
     561          38.30            75.19       439.6           0.09267   
     562          42.79           128.70       915.0           0.14170   
     563          29.41           179.10      1819.0           0.14070   
     564          26.40           166.10      2027.0           0.14100   
     565          38.25           155.00      1731.0           0.11660   
     566          34.12           126.70      1124.0           0.11390   
     567          39.42           184.60      1821.0           0.16500   
     568          30.37            59.16       268.6           0.08996   
     
          worst compactness  worst concavity  worst concave points  worst symmetry  \
     0              0.66560          0.71190               0.26540          0.4601   
     1              0.18660          0.24160               0.18600          0.2750   
     2              0.42450          0.45040               0.24300          0.3613   
     3              0.86630          0.68690               0.25750          0.6638   
     4              0.20500          0.40000               0.16250          0.2364   
     5              0.52490          0.53550               0.17410          0.3985   
     6              0.25760          0.37840               0.19320          0.3063   
     7              0.36820          0.26780               0.15560          0.3196   
     8              0.54010          0.53900               0.20600          0.4378   
     9              1.05800          1.10500               0.22100          0.4366   
     10             0.15510          0.14590               0.09975          0.2948   
     11             0.56090          0.39650               0.18100          0.3792   
     12             0.39030          0.36390               0.17670          0.3176   
     13             0.19240          0.23220               0.11190          0.2809   
     14             0.77250          0.69430               0.22080          0.3596   
     15             0.65770          0.70260               0.17120          0.4218   
     16             0.18710          0.29140               0.16090          0.3029   
     17             0.42330          0.47840               0.20730          0.3706   
     18             0.31500          0.53720               0.23880          0.2768   
     19             0.17730          0.23900               0.12880          0.2977   
     20             0.27760          0.18900               0.07283          0.3184   
     21             0.11480          0.08867               0.06227          0.2450   
     22             0.59540          0.63050               0.23930          0.4667   
     23             0.26000          0.31550               0.20090          0.2822   
     24             0.35780          0.46950               0.20950          0.3613   
     25             0.39490          0.38530               0.25500          0.4066   
     26             0.66430          0.55390               0.27010          0.4264   
     27             0.21170          0.34460               0.14900          0.2341   
     28             0.61100          0.63350               0.20240          0.4027   
     29             0.28120          0.24890               0.14560          0.2756   
     ..                 ...              ...                   ...             ...   
     539            0.30640          0.33930               0.05000          0.2790   
     540            0.21180          0.17970               0.06918          0.2329   
     541            0.42020          0.40400               0.12050          0.3187   
     542            0.13760          0.16110               0.10950          0.2722   
     543            0.13810          0.10620               0.07958          0.2473   
     544            0.20370          0.13770               0.06845          0.2249   
     545            0.15170          0.10490               0.07174          0.2642   
     546            0.08842          0.04384               0.02381          0.2681   
     547            0.22460          0.17830               0.08333          0.2691   
     548            0.09546          0.09350               0.03846          0.2552   
     549            0.16330          0.06194               0.03264          0.3059   
     550            0.07348          0.00000               0.00000          0.2458   
     551            0.17820          0.15640               0.06413          0.3169   
     552            0.10640          0.08653               0.06498          0.2407   
     553            0.08298          0.07993               0.02564          0.2435   
     554            0.16200          0.24390               0.06493          0.2372   
     555            0.17100          0.20000               0.09127          0.2226   
     556            0.12000          0.01005               0.02232          0.2262   
     557            0.07158          0.00000               0.00000          0.2475   
     558            0.31710          0.36620               0.11050          0.2258   
     559            0.25170          0.36300               0.09653          0.2112   
     560            0.22640          0.13260               0.10480          0.2250   
     561            0.05494          0.00000               0.00000          0.1566   
     562            0.79170          1.17000               0.23560          0.4089   
     563            0.41860          0.65990               0.25420          0.2929   
     564            0.21130          0.41070               0.22160          0.2060   
     565            0.19220          0.32150               0.16280          0.2572   
     566            0.30940          0.34030               0.14180          0.2218   
     567            0.86810          0.93870               0.26500          0.4087   
     568            0.06444          0.00000               0.00000          0.2871   
     
          worst fractal dimension  
     0                    0.11890  
     1                    0.08902  
     2                    0.08758  
     3                    0.17300  
     4                    0.07678  
     5                    0.12440  
     6                    0.08368  
     7                    0.11510  
     8                    0.10720  
     9                    0.20750  
     10                   0.08452  
     11                   0.10480  
     12                   0.10230  
     13                   0.06287  
     14                   0.14310  
     15                   0.13410  
     16                   0.08216  
     17                   0.11420  
     18                   0.07615  
     19                   0.07259  
     20                   0.08183  
     21                   0.07773  
     22                   0.09946  
     23                   0.07526  
     24                   0.09564  
     25                   0.10590  
     26                   0.12750  
     27                   0.07421  
     28                   0.09876  
     29                   0.07919  
     ..                       ...  
     539                  0.10660  
     540                  0.08134  
     541                  0.10230  
     542                  0.06956  
     543                  0.06443  
     544                  0.08492  
     545                  0.06953  
     546                  0.07399  
     547                  0.09479  
     548                  0.07920  
     549                  0.07626  
     550                  0.06592  
     551                  0.08032  
     552                  0.06484  
     553                  0.07393  
     554                  0.07242  
     555                  0.08283  
     556                  0.06742  
     557                  0.06969  
     558                  0.08004  
     559                  0.08732  
     560                  0.08321  
     561                  0.05905  
     562                  0.14090  
     563                  0.09873  
     564                  0.07115  
     565                  0.06637  
     566                  0.07820  
     567                  0.12400  
     568                  0.07039  
     
     [569 rows x 30 columns], 0      0.0
     1      0.0
     2      0.0
     3      0.0
     4      0.0
     5      0.0
     6      0.0
     7      0.0
     8      0.0
     9      0.0
     10     0.0
     11     0.0
     12     0.0
     13     0.0
     14     0.0
     15     0.0
     16     0.0
     17     0.0
     18     0.0
     19     1.0
     20     1.0
     21     1.0
     22     0.0
     23     0.0
     24     0.0
     25     0.0
     26     0.0
     27     0.0
     28     0.0
     29     0.0
           ... 
     539    1.0
     540    1.0
     541    1.0
     542    1.0
     543    1.0
     544    1.0
     545    1.0
     546    1.0
     547    1.0
     548    1.0
     549    1.0
     550    1.0
     551    1.0
     552    1.0
     553    1.0
     554    1.0
     555    1.0
     556    1.0
     557    1.0
     558    1.0
     559    1.0
     560    1.0
     561    1.0
     562    0.0
     563    0.0
     564    0.0
     565    0.0
     566    0.0
     567    0.0
     568    1.0
     Name: target, dtype: float64)



### Question 4
Using `train_test_split`, split `X` and `y` into training and test sets `(X_train, X_test, y_train, and y_test)`.

**Set the random number generator state to 0 using `random_state=0` to make sure your results match the autograder!**

*This function should return a tuple of length 4:* `(X_train, X_test, y_train, y_test)`*, where* 
* `X_train` *has shape* `(426, 30)`
* `X_test` *has shape* `(143, 30)`
* `y_train` *has shape* `(426,)`
* `y_test` *has shape* `(143,)`


```python
from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    seed = 0 
    X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=seed)

    return X_train, X_test, y_train, y_test
```

### Question 5
Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with `X_train`, `y_train` and using one nearest neighbor (`n_neighbors = 1`).

*This function should return a * `sklearn.neighbors.classification.KNeighborsClassifier`.


```python
from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    
    
    return knn.fit(X_train, y_train)

answer_five()
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=1, p=2,
               weights='uniform')



### Question 6
Using your knn classifier, predict the class label using the mean value for each feature.

Hint: You can use `cancerdf.mean()[:-1].values.reshape(1, -1)` which gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).

*This function should return a numpy array either `array([ 0.])` or `array([ 1.])`*


```python
def answer_six():
    knn = KNeighborsClassifier(n_neighbors = 1)
    X_train, X_test, y_train, y_test = answer_four()
    knn.fit(X_train, y_train)
    
    
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    
    features_label = knn.predict(means)
    
    
    
    a = np.array([features_label[0].astype('float')])
    
    return a


answer_six()
```




    array([ 1.])



### Question 7
Using your knn classifier, predict the class labels for the test set `X_test`.

*This function should return a numpy array with shape `(143,)` and values either `0.0` or `1.0`.*


```python
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    testX_label = [] 
    
    for i in range(len(X_test)):
        x = knn.predict(X_test.iloc[i].values.reshape(1,-1))
        testX_label.append(x[0].astype('float'))
    
    
    
    return np.asarray(testX_label)


answer_seven()
```




    array([ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,
            1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,
            1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,
            0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,
            0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,
            1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,
            1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,
            0.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            0.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,
            0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.])



### Question 8
Find the score (mean accuracy) of your knn classifier using `X_test` and `y_test`.

*This function should return a float between 0 and 1*


```python
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score = knn.score(X_test, y_test)
    
    return score


answer_eight()
```




    0.91608391608391604



### Optional plot

Try using the plotting function below to visualize the differet predicition scores between training and test sets, as well as malignant and benign cells.


```python
def accuracy_plot():
    import matplotlib.pyplot as plt

    %matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
```

Uncomment the plotting function to see the visualization.

**Comment out** the plotting function when submitting your notebook for grading. 


```python
#accuracy_plot() 
```


```python

```
