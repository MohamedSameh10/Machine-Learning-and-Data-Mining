import math
import pandas as pd
from operator import itemgetter

class DecisionTree:
    def __init__(self, df, target, positive,parent_val, parent):
        self.data = df
        self.target = target
        self.positive = positive
        self.parent_val = parent_val
        self.parent = parent
        self.children = []
        self.decision = ''

    def getEntropy(self, data):
        p=sum(data[self.target] == self.positive)
        n = data.shape[0] - p
        entropy_p = -(p/(p+n)) * math.log2(p/(p+n)) if (p/(p+n)) != 0 else 0
        entropy_n = - (1-(p/(p+n))) * math.log2(1-(p/(p+n))) if (1-(p/(p+n))) != 0 else 0
        return entropy_p + entropy_n

    def getGain(self, feature):
        avg_info = 0
        for val in self.data[feature].unique():
            avg_info += self.getEntropy(self.data[self.data[feature] == val]) * sum(self.data[feature] == val) / self.data.shape[0]
        return self.getEntropy(self.data) - avg_info

    def updateTree(self):
        self.features = [col for col in self.data.columns if col != self.target]
        self.entropy = self.getEntropy(self.data)
        if self.entropy != 0:
            self.gains = [(feature, self.getGain(feature)) for feature in self.features]
            self.splitter = max(self.gains, key=itemgetter(1))[0]
            remainingColumns = [k for k in self.data.columns if k != self.splitter]
            for val in self.data[self.splitter].unique():
                df_tmp = self.data[self.data[self.splitter] == val][remainingColumns]
                tmp_node = DecisionTree(df_tmp, self.target, self.positive, val, self.splitter)
                tmp_node.updateTree()
                self.children.append(tmp_node)

def print_tree(n):
    for child in n.children:
        if child:
            print(child.__dict__.get('parent', ''))
            print(child.__dict__.get('parent_val', ''), '\n')
            print_tree(child)

#For the assignment dataset
"""
dataset = pd.read_csv("ass3.csv")
tree = DecisionTree(dataset, 'A', 1, '', '')
tree.updateTree()
print_tree(tree)
"""
#For the Cardiac dataset

dataset = pd.read_csv("cardio_train.csv",sep=";")
tree = DecisionTree(dataset, 'cardio', 1, '', '')
tree.updateTree()
print_tree(tree)
