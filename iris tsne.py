####################
# load iris
####################

from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

####################
# tsne
####################

from sklearn.manifold import TSNE

tsne = TSNE( n_components=2,
             learning_rate='auto',
             init='random' )
x_tsne = tsne.fit_transform(x)

####################
# visualization
####################

import matplotlib.pyplot as plt

# different class have different color
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y) # marker='.'
plt.show()
