# Association - FP Grwoth Algoritm


```python
#importing module
import pandas as pd
```
This line imports the pandas library under the alias 'pd'. Pandas is commonly used for data manipulation and analysis in Python.

```python
# dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv")
```
This line reads a CSV file named "Market_Basket_Optimisation.csv" into a pandas DataFrame called `dataset`.

```python
# printing the shape of the dataset
dataset.shape
```
This line prints the shape of the dataset, which represents the number of rows and columns in the DataFrame.

```python
# printing the columns and few rows using head
dataset.head()
```
This line prints the first few rows of the dataset along with the column names, providing an overview of the data.

```python
# Generating numpy transactions array
import numpy as np
```
This line imports the numpy library under the alias 'np'. Numpy is commonly used for numerical computing in Python.

```python
# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transaction.append(dataset.values[i,j])
```
These lines loop through each row and column of the DataFrame `dataset` and append each item to the `transaction` list.

```python
# converting to numpy array
transaction = np.array(transaction)
print(transaction)
print(len(transaction))
```
This section converts the `transaction` list into a numpy array and prints it along with its length.

```python
# Top 5 items
#  Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"]) 
# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
df["incident_count"] = 1 
#  Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)
# Making a New Appropriate Pandas DataFrame for Visualizations  
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
#  Initial Visualizations
df_table.head(5).style.background_gradient(cmap='Blues')
```
This part of the code creates a DataFrame `df` from the `transaction` array, adds a column 'incident_count' and counts occurrences of each item. It then visualizes the top 5 items using a gradient style.


```python
# Tree map
# importing required module
import plotly.express as px
```
This part imports Plotly Express library as `px`, which is a high-level interface for creating various types of interactive plots.

```python
# to have a same origin
df_table["all"] = "Top 50 items" 
```
This line adds a new column named "all" to the DataFrame `df_table` and assigns the value "Top 50 items" to all rows. This column will be used to provide a common origin for the tree map visualization.

```python
# creating tree map using plotly
fig = px.treemap(df_table.head(50), path=['all', "items"], values='incident_count',
                  color=df_table["incident_count"].head(50), hover_data=['items'],
                  color_continuous_scale='Blues',
                )
```
This code creates a tree map visualization using Plotly Express (`px.treemap()`). It takes the DataFrame `df_table` and specifies the hierarchical path for each item, the values (incident counts) to be represented by the size of the rectangles, and the color scale.

```python
# ploting the treemap
fig.show()
```

```python
# Pre-processing dataset
# Transform Every Transaction to Seperate List & Gather Them into Numpy Array
transaction = []
for i in range(dataset.shape[0]):
    transaction.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])
# creating the numpy array of the transactions
transaction = np.array(transaction)
# importing the required module
from mlxtend.preprocessing import TransactionEncoder
# initializing the transactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)
# dataset after encoded
dataset.head()
```
This section preprocesses the dataset for association rule mining. It transforms each transaction into a separate list and gathers them into a numpy array. Then, it uses the `TransactionEncoder` from `mlxtend.preprocessing` to encode the transactions into a binary format suitable for association rule mining. Finally, it converts the encoded transactions into a pandas DataFrame named `dataset`.

```python
# Top 30 items
# select top 30 items
first30 = df_table["items"].head(30).values 
# Extract Top 30
dataset = dataset.loc[:,first30] 
# shape of the dataset
dataset.shape
```
This part selects the top 30 items based on their occurrence frequency from the `df_table` DataFrame. It then extracts only those top 30 items from the `dataset` DataFrame. Finally, it prints the shape of the resulting dataset to ensure that only the top 30 items are retained.

```python
# Implementing FP growth Algorithm
#Importing Libraries
from mlxtend.frequent_patterns import fpgrowth
#running the fpgrowth algorithm
res=fpgrowth(dataset,min_support=0.05, use_colnames=True)
# printing top 10
res.head(10)
```
This section implements the FP-Growth algorithm for frequent itemset mining using the `fpgrowth` function from `mlxtend.frequent_patterns`. It sets the minimum support threshold to 0.05 and specifies to use column names from the dataset. It then prints the top 10 frequent itemsets along with their support values.

```python
# Association rules
# importing required module
from mlxtend.frequent_patterns import association_rules
# creating asssociation rules
res=association_rules(res, metric="lift", min_threshold=1)
# printing association rules
res
```
This part generates association rules from the frequent itemsets using the `association_rules` function from `mlxtend.frequent_patterns`. It specifies the metric as "lift" and the minimum threshold as 1. It then prints the resulting association rules.

```python
# Sorting
# Sort values based on confidence
res.sort_values("confidence",ascending=False)
```
Finally, this section sorts the association rules based on confidence in descending order and prints the sorted rules.



# Classification - Support Vector Machine Algorithm



### 1. Importing Libraries:
```python
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
```
- **numpy (np)**: Numpy is used for numerical operations and array manipulations.
- **matplotlib.pyplot (plt)**: Matplotlib is a plotting library used to visualize data and results.
- **pandas (pd)**: Pandas is used for data manipulation and analysis.

### 2. Loading Dataset:
```python
data_set = pd.read_csv('Social_Network_Ads.csv')
```
- The dataset 'Social_Network_Ads.csv' is loaded into a Pandas DataFrame named `data_set`.

### 3. Data Preprocessing:
```python
x = data_set.iloc[:, [2,3]].values  
y = data_set.iloc[:, 4].values  
```
- The independent variables (features) are extracted into the variable `x`, containing columns 2 and 3 of the dataset.
- The dependent variable (target) is extracted into the variable `y`, containing column 4 of the dataset.

### 4. Splitting the Dataset:
```python
from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
```
- The dataset is split into training and testing sets using a 75%-25% split ratio.
- `x_train` and `y_train` contain the features and labels for the training set, respectively.
- `x_test` and `y_test` contain the features and labels for the testing set, respectively.

### 5. Feature Scaling:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```
- Feature scaling is performed to standardize the range of features.
- `StandardScaler()` is used to scale features to have a mean of 0 and a standard deviation of 1.
- `fit_transform()` is applied to the training set, and `transform()` is applied to the testing set using the same scaling parameters learned from the training set.

### 6. Training the SVM Classifier:
```python
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train)
```
- An SVM classifier is initialized with a linear kernel and a random seed for reproducibility.
- The classifier is trained on the scaled training data using the `fit()` method.

### 7. Prediction:
```python
y_pred = classifier.predict(x_test)
```
- The trained SVM classifier is used to predict the labels for the test set using the `predict()` method.

### 8. Evaluation:
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
```
- The confusion matrix is computed to evaluate the performance of the classifier.

### 9. Visualization:
```python
from matplotlib.colors import ListedColormap  

x_set, y_set = x_train, y_train  

x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),  
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))  

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
             alpha=0.2, cmap=ListedColormap(('red', 'green')))  

plt.xlim(x1.min(), x1.max()) 
plt.ylim(x2.min(), x2.max()) 

for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                c=ListedColormap(('red', 'green'))(i), label=j)  

plt.title('SVM classifier (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  
```
- Meshgrid is created to generate a grid of points for visualization.
- The decision boundary is plotted based on the SVM classifier's predictions using `contourf()`.
- Data points from the training set are plotted with different colors based on their class labels using `scatter()`.
- Plot title, labels, and legend are added for better understanding.


# Clustering - DBSCAN Algorithm

Sure, let's break down the code step by step:

1. **Imports**: 
    ```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    ```

    - These lines import necessary libraries for data manipulation, visualization, and clustering. 
    - `numpy` and `pandas` are used for data manipulation.
    - `seaborn` and `matplotlib.pyplot` are used for visualization.
    - `DBSCAN` is imported from scikit-learn, which is a clustering algorithm.

2. **Reading the Dataset**:
    ```python
    data = pd.read_csv('Clustering_gmm.csv')
    ```

    - This line reads a CSV file named `'Clustering_gmm.csv'` and loads it into a pandas DataFrame called `data`.

3. **Previewing the Dataset**:
    ```python
    data.head()
    ```

    - This displays the first few rows of the dataset to get an idea of its structure.

4. **Extracting Features**:
    ```python
    X_train = data[['Weight', 'Height']]
    ```

    - This line extracts the features from the dataset. It assumes that the dataset has columns named `'Weight'` and `'Height'`.

5. **DBSCAN Clustering**:
    ```python
    clustering = DBSCAN(eps=12.5, min_samples=4).fit(X_train)
    ```

    - This line applies the DBSCAN clustering algorithm to the extracted features.
    - `eps` is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - `min_samples` is the number of samples in a neighborhood for a point to be considered as a core point.

6. **Creating Dataset with Clusters**:
    ```python
    DBSCAN_dataset = X_train.copy()
    DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_
    ```

    - This creates a copy of the original feature dataset and adds a new column `'Cluster'` to it, which contains the cluster labels assigned by DBSCAN.

7. **Counting Cluster Labels**:
    ```python
    DBSCAN_dataset.Cluster.value_counts().to_frame()
    ```

    - This counts the number of points in each cluster and presents the result as a DataFrame.

8. **Identifying Outliers**:
    ```python
    outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster']==-1]
    ```

    - This selects the points labeled as outliers (those with cluster label `-1`).

9. **Plotting Clusters**:
    ```python
    fig2, (axes) = plt.subplots(1,2,figsize=(12,5))
    
    sns.scatterplot('Weight', 'Height',
                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1],
                    hue='Cluster', ax=axes[0], palette='Set2', legend='full', s=200)
    
    axes[0].scatter(outliers['Weight'], outliers['Height'], s=10, label='outliers', c="k")
    axes[0].legend()
    plt.setp(axes[0].get_legend().get_texts(), fontsize='12')
    plt.show()
    ```

    - This code plots the clusters and outliers.
    - It creates a scatter plot of the dataset, coloring points based on their cluster labels.
    - Outliers are marked separately with black dots.
    - The legend is included to differentiate clusters.
    - Finally, it displays the plot.


# Similarity - Hamming Distance

1. **Hamming Distance Function (`def hamming_distance(str1, str2)`)**:
   - This function takes two strings, `str1` and `str2`, as input parameters.
   - It first checks if the lengths of the input strings are equal. If they are not equal, it raises a `ValueError` indicating that the strings must be of equal length to calculate the Hamming distance.
   - Then, it initializes a variable `distance` to 0, which will be used to count the number of differing characters.
   - The function iterates through the characters of both strings simultaneously using the `zip()` function. This allows it to compare characters at the same position in both strings.
   - For each pair of characters, if they are not equal, it increments the `distance` variable.
   - Finally, it returns the calculated Hamming distance.

2. **Reading Strings from a Text File**:
   - The code uses a context manager (`with open("dataset.txt", "r") as file:`) to open the file named "dataset.txt" in read mode. Using a context manager ensures that the file is properly closed after its suite finishes execution.
   - It then reads the contents of the file using the `read()` method, which reads the entire contents of the file as a single string.
   - The `splitlines()` method is called on the string to split it into a list of strings, where each string represents a line from the file. This is done to process each line of the file as a separate string.

3. **Calculating Hamming Distance for Combinations**:
   - The code iterates through all possible combinations of strings from the list obtained in the previous step using nested loops.
   - It uses nested `for` loops to generate pairs of indices `i` and `j` where `i` is less than `j`. This ensures that each pair is considered only once and avoids duplicate calculations.
   - For each pair of strings, it retrieves the strings `string_a` and `string_b` from the list of strings.
   - To calculate the Hamming distance, it first determines the maximum length among the lengths of `string_a` and `string_b`. This is necessary because Hamming distance requires strings of equal length. It pads both strings with spaces using the `ljust()` method so that they become of equal length.
   - After padding, it calculates the Hamming distance between the padded strings using the `hamming_distance()` function.
   - Finally, it prints the Hamming distance along with the pair of strings to the console.

This code effectively reads strings from a file, computes the Hamming distance for all pairs of strings, and prints the results. Hamming distance is a measure of the difference between two strings of equal length and is often used in various applications such as error detection and correction algorithms, DNA sequence analysis, and similarity measures in natural language processing.
