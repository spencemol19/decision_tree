from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
# from numpy import ndarray

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    iris = load_iris()

    # Model (can also use single decision tree)
    model = RandomForestClassifier(n_estimators=10)

    # print(iris.data)

    # print(iris.feature_names)

    # print(iris.target)

    # print(iris.target_names)

    # Train
    model.fit(iris.data, iris.target)
    
    # Extract single tree
    estimator = model.estimators_[5]

    # model.predict()
    
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot', 
                    feature_names = iris.feature_names,
                    class_names = iris.target_names,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    img = mpimg.imread('tree.png')
    imgplot = plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()