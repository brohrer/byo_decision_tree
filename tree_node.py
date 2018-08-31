import numpy as np


class TreeNode(object):
    """
    The foundational data structure for DecisionTree.
    Root, branches, and leaves are all TreeNodes.
    """
    def __init__(
        self,
        features=None,
        parent=None,
        recommendation=None,
        split_feature=None,
    ):
        """
        Parameters
        ----------
        features: list
            The feature values corresponding to this node: 1 or 0.
            None if that feature hasn't been split on.
        parent: TreeNode
            If None, then this is the root TreeNode.
        recommendation: float
            The departure time recommendation for this node.
        split_feature: int
            The index of the feature on which
            this node's children are split.
        """
        self.parent = parent
        # If/when this node gets split, it will get a hi and low branch.
        self.hi_branch = None
        self.lo_branch = None
        self.is_leaf = True
        # The feature index on which this node is split to create
        # its children. If this node is a leaf, this will be None.
        self.split_feature = split_feature
        # The list of all feature values associated with this node.
        # Features that haven't been split on yet are given as None.
        self.features = features
        # The departure time recommendation associated with this node.
        self.recommendation = recommendation

    def attempt_split(self, data, err_fn, n_min):
        """
        Try to split this node into two child nodes.

        Parameters
        ----------
        data: DataFrame
            Features for each of the data points
        err_fn: function
            The error function that determines the fitness of a split.
            Choose a split that minimizes the combined error of the
            two resulting branches.
        n_min: int
            The number of data points that need to remain in a node to
            make it viable.

        Returns
        -------
        success: boolean
            If a split happened, returns True.
        """
        success = False

        n_features = len(data.columns)
        # Initialize the root node properly.
        if self.features is None:
            self.features = [None] * n_features

        node_data = self.find_members(data)

        feature_candidates = [
            i for i, j in enumerate(self.features) if j is None]
        best_feature = -1
        best_split_score = 1e10
        best_hi_recommendation = self.recommendation
        best_lo_recommendation = self.recommendation

        # Check each of the potential feature dimensions
        # for split quality.
        if len(feature_candidates) > 0:
            # Shuffe these up to prevent order artifacts in ties.
            np.random.shuffle(feature_candidates)
            for i_feature in feature_candidates:
                hi_data = node_data.loc[node_data.iloc[:, i_feature] == 1, :]
                lo_data = node_data.loc[node_data.iloc[:, i_feature] == 0, :]
                # Only consider splits for which both child nodes
                # maintain more than the minimum number of data points.
                if hi_data.shape[0] >= n_min and lo_data.shape[0] >= n_min:
                    hi_score, hi_recommendation = err_fn(list(hi_data.index))
                    lo_score, lo_recommendation = err_fn(list(lo_data.index))
                    split_score = hi_score + lo_score
                    # Track the best-so-far solution
                    if split_score < best_split_score:
                        best_split_score = split_score
                        best_feature = i_feature
                        best_hi_recommendation = hi_recommendation
                        best_lo_recommendation = lo_recommendation
                        success = True

            # Act on the best split.
            if success:
                # Make a copy of the features list.
                hi_features = list(self.features)
                hi_features[best_feature] = 1
                self.hi_branch = TreeNode(
                    parent=self,
                    features=hi_features,
                    recommendation=best_hi_recommendation,
                )

                lo_features = list(self.features)
                lo_features[best_feature] = 0
                self.lo_branch = TreeNode(
                    parent=self,
                    features=lo_features,
                    recommendation=best_lo_recommendation,
                )

                self.is_leaf = False
                self.split_feature = best_feature

        return success

    def find_members(self, data):
        """
        Find all of the dates within features that belong to this node.

        Parameters
        ----------
        data: DataFrame
            A set of dates under consideration
            and their associated features.

        Returns:
        --------
        member_data: DataFrame
            A subset of the rows of the features DataFrame.
        """
        member_data = data
        for i_feature, feature in enumerate(self.features):
            if self.features[i_feature] is not None:
                member_data = member_data.loc[
                        member_data.iloc[:, i_feature] == feature]

        return member_data
