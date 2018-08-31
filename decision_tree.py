import numpy as np
import matplotlib.pyplot as plt

from tree_node import TreeNode


class DecisionTree(object):
    """
    A decision tree for data with binary features, that is, where each
    feature has a value of either 0 or 1.
    """
    def __init__(self, err_fn=None, n_min=20, debug=False):
        """
        Parameters
        ----------
        err_fn: function
           This is the function that will be used to judge the fitness
           of each leaf of the tree. It has this signature:

           error = err_fun(datestrs)

           where datestrs is a list of strings of YYYY-MM-DD format and
           error is a float.

        n_min: int
            The minimum number of members a leaf node is permitted to have.
        debug: boolean
        """
        self.debug = debug
        self.n_min = n_min
        if err_fn is None:
            raise ValueError("An err_fn keyword argument must be supplied.")
        else:
            self.err_fn = err_fn

        # Initialize the root of the tree.
        self.root = TreeNode()

        # feature_names are a list of strings associated with the features.
        # They will be assigned during training.
        # They are included for ease of interpreting the tree.
        self.feature_names = None

    def train(self, training_features):
        """
        Recursively split nodes of the tree until
        they can't be split any more.

        Parameters
        ----------
        training_features: DataFrame
        """
        self.feature_names = training_features.columns
        nodes_to_check = [self.root]

        while len(nodes_to_check) > 0:
            current_node = nodes_to_check.pop()
            success = current_node.attempt_split(
                training_features, self.err_fn, self.n_min)
            if success:
                nodes_to_check.append(current_node.lo_branch)
                nodes_to_check.append(current_node.hi_branch)
        return

    def estimate(self, features):
        """
        Parameters
        ----------
        features: list of floats
            Feature values for the date for which to make a recommendation.
            Values can be 0 or 1.

        Returns
        -------
        recommended_departure: int
            The recommended departure time,
            in minutes from when work starts.
            (Negative means minutes before.)
        """
        if len(features) != len(self.root.features):
            if self.debug:
                print('The feature you are asking to estimate has a')
                print('different number of features than the tree.')
            return None

        current_node = self.root
        while True:
            if current_node.is_leaf:
                return current_node.recommendation
            if features[current_node.split_feature] == 0:
                current_node = current_node.lo_branch
            elif features[current_node.split_feature] == 1:
                current_node = current_node.hi_branch
            else:
                if self.debug:
                    print('Feature', current_node.split_feature)
                    print('is not 0 or 1. Something is wrong.')
                return None

    def render(self):
        """
        Create a plot that describes the tree.
        """
        plt.figure(34857)
        plt.clf()

        def plot_node(node, level, x):
            """
            Render the important information about a single node.

            Parameters
            ----------
            node: TreeNode
            level: int
            x: float
                The x position of the center of this node's info.
            """
            recommendation = node.recommendation
            feature_name = self.feature_names[node.split_feature]
            if node.is_leaf:
                node_text = 'at: {rec}\n'.format(rec=recommendation)
            else:
                node_text = ''.join([
                    'at: {rec}\n'.format(rec=recommendation),
                    '{feature_name}?\n'.format(feature_name=feature_name),
                    'no    yes'])
            plt.text(
                x, -level,
                node_text,
                horizontalalignment='center',
                verticalalignment='center',
            )
            return

        def plot_branches(level, x0, y_delta=.2):
            """
            Draw the branches between the current node and its children.

            Parameters
            ----------
            level: int
            x: float
            y_delta: float
                The amount of y-tail to be hidden.

            Returns
            -------
            x_lo, x_hi: floats
                The x positions of the low and high branch nodes.
            """
            y0 = -level
            y3 = -level - 1
            y1 = y0 - y_delta
            y2 = y3 + y_delta

            x3_lo = x0 - 2 ** y3
            x3_hi = x0 + 2 ** y3
            slope_lo = 1 / (x3_lo - x0)
            slope_hi = 1 / (x3_hi - x0)
            x_lo_delta = y_delta / slope_lo
            x_hi_delta = y_delta / slope_hi
            x1_lo = x0 + x_lo_delta
            x1_hi = x0 + x_hi_delta
            x2_lo = x3_lo - x_lo_delta
            x2_hi = x3_hi - x_hi_delta

            plt.plot([x1_lo, x2_lo], [y1, y2], color='black')
            plt.plot([x1_hi, x2_hi], [y1, y2], color='black')

            return x3_lo, x3_hi

        def recurse(node, level, x):
            """
            Cycle through the tree in a depth-first manner.

            Parameters
            ----------
            node: TreeNode
            level: int
            x: float
            """
            plot_node(node, level, x)
            if node.is_leaf:
                return

            x_lo, x_hi = plot_branches(level, x)
            recurse(node.lo_branch, level + 1, x_lo)
            recurse(node.hi_branch, level + 1, x_hi)
            return

        initial_level = 0
        initial_x = 0
        recurse(self.root, initial_level, initial_x)
        plt.show()
