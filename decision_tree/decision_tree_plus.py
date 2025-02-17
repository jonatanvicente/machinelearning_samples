import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = sns.load_dataset("iris") # Datasets are located on https://github.com/mwaskom/seaborn-data
df.head()

print(df.species.unique()) # Different species of iris

'''
Now we want to predict the species of iris based on the other columns,
using a Decision Tree Classifier.
'''
# Independent variables: all columns except "species"
X = df.drop("species", axis=1)

le = LabelEncoder() # species contains strings, we need to encode them to numbers
especies = le.fit_transform(df["species"]) # Output: Array with encoded species, values 0, 1, 2
y = especies # Dependent variable: "species"

# slice the data into training and testing sets

'''
random_state is used to ensure reproducibility of results. By setting a specific integer value for random_state, 
you ensure that the random processes (such as shuffling the data) produce the same results every time you run the code
'''
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X,
                                                           y,
                                                           train_size=0.8,
                                                           random_state=42)
# Object with the Decision Tree Classifier
arbol = DecisionTreeClassifier()
arbol.fit(X_entrena, y_entrena)

#  A bit more big and clear
plt.figure(figsize=(20, 20))
plot_tree(decision_tree=arbol);
# plt.show()

# Better for interactive visualization
plt.figure(figsize=(20, 20))
plot_tree(decision_tree=arbol, filled=True);
# plt.show()

# to read the names
plt.figure(figsize=(20, 20))
plot_tree(decision_tree=arbol,
          class_names=["setosa", "versicolor", "virginica"],
          filled=True);
#plt.show()

# to see the decisions made by the tree
plt.figure(figsize=(20, 20))
plot_tree(decision_tree=arbol,
          class_names=["setosa", "versicolor", "virginica"],
          feature_names = df.columns.to_list(),
          filled=True);
plt.show()

'''
The model has identified that the best way to start asking questions to classify our flowers is to ask ourselves if the length 
of its petal is equal to or less than 2.45, because if so, it is highly likely that it is a flower of the setosa species.

gini = 0.667: 
    The Gini coefficient measures the impurity of the node, and therefore how reliable it is. 
    If it had a value of 0 it would indicate that it is a perfectly pure node (and that all 
    the samples in the node are of a single class). In this case we have a value of 0.667, 
    which indicates a certain level of class mixing in this node.

samples = 120: 
    This indicates how many samples have passed through this node. 
    Since this is the root node, naturally all 120 samples in the set passed through this node.

value = [40, 41, 39]: 
    This array shows how many samples of each class are present at this node. 
    The first position in the array corresponds to the first class ('setosa'), 
    the second to the second class ('versicolor'), 
    and the third to the third class ('virginica'). 
    So there are 40 samples of 'setosa', 41 of 'versicolor', and 39 of 'virginica'.

class = versicolor: 
    This is the result of the classification at this node. 
    Since 'versicolor' is the class with the most samples at this node (albeit by a very small margin), 
    any sample that reaches this point and does not split further will be classified as 'versicolor'.

In the box on the left we have 40 samples that met the condition of having a petal length equal to or less than 2.45, 
    all of which correspond to the setosa species, and as you can see, this node has a gini of 0, so it is absolutely reliable.

The other 80 samples fell into a new node, where Machine Learning has identified that the best step to continue classifying 
    the flowers is to verify whether the length of their petals is less than or equal to 4.75. 
    If this is so or not, it will distribute each case along a series of new questions, until it identifies which species they belong to.

As you can see, each of the end leaves of our tree has a perfect gini
We can trust that our model has learned to classify iris flowers with absolute precision

This level of perfection is due to the fact that we have used a fictitious and very controlled database. 
'''


