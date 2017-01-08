import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils import check_random_state
from sklearn import metrics, ensemble, tree, linear_model
from sklearn.model_selection import train_test_split


def random_point_on_sphere(dim=2, random_state=None):
    """
    Return a random point on the sphere, uniformly selected with respect to volume.
    """
    random_state = check_random_state(random_state)
    pt = 0
    norm = 0
    while norm < 1e-12:
        pt = random_state.normal(size=dim)
        norm = np.linalg.norm(pt)
    return pt / norm


def random_son(dim=2, random_state=None):
    """
    Return a random element of SO(dim).

    SO(dim) in this case consists of the orthogonal matrices with determinant 1.
    Matrices are selected uniformly with respect to Haar measure.
    """
    random_state = check_random_state(random_state)
    result = -np.eye(dim, dtype=np.float)

    result[0, 0] = 1

    for i in range(2, dim+1):
        x = random_point_on_sphere(dim=i, random_state=random_state)
        result[:i, :i] = \
            result[:i, :i] - (np.dot(2*x, result[:i, :i])[:, np.newaxis]*x).T

    return result


class RandomCoordinateChanger(BaseEstimator, TransformerMixin):
    """
    Choose a random change of coordinates and apply to X.
    
    Parameters:

    usecols - a sequence of integers indicating the columns of X on which
    to apply the coordinate change. If None, every column of X will be used.

    transform_dimension - the dimension of the coordinate change. If this
    is less than len(usecols), for each coordinate change a random selection
    of transform_dimension features will be selected to apply the coordinate
    change. If None or greater than len(usecols), the dimension of the
    coordinate change will be len(usecols). The only reason for this to be
    less than len(usecols) is performance.
    
    transform_repeats - the number of random coordinate changes to select
    and apply. If transform_dimension is len(usecols), there is no reason
    for this to be greater than 1. If transform_repeats is 'auto',
    transform_dimension // len(usecols) will be used.
    
    copy - whether to make a copy of X (otherwise it will be transformed in
    place).
    
    random_state - int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator. If
    RandomState instance, random_state is the random number generator. If None,
    the random number generator is the RandomState instance used by np.random.
    """
    def __init__(self, transform_dimension=None, transform_repeats='auto', copy=True,
                 usecols=None, random_state=None):
        self.transform_dimension = transform_dimension
        self.transform_repeats = transform_repeats
        self.copy = copy
        self.usecols = usecols
        self.random_state = random_state

    def fit(self, X, y=None):
        self.random_state = check_random_state(self.random_state)

        if self.usecols is None:
            indices = np.array(list(range(X.shape[1])))
        else:
            indices = np.array(self.usecols)

        X = np.array(X)
        m = len(indices)
        if self.transform_dimension is None:
            self.real_dimension_ = m
        else:
            self.real_dimension_ = np.min(self.transform_dimension, m)
        if self.transform_repeats == 'auto':
            self.real_repeats_ = m // self.real_dimension_
        else:
            self.real_repeats_ = self.transform_repeats

        def create_son():
            return random_son(dim=self.real_dimension_,
                              random_state=self.random_state)
        self.transforms = np.array([create_son()
                                    for i in range(self.real_repeats_)])

        def choose_indices():
            return self.random_state.choice(indices,
                                            size=self.real_dimension_,
                                            replace=False)
        self.indices_ = np.array([choose_indices()
                                 for i in range(self.real_repeats_)])
        return self

    def transform(self, X):
        X = np.array(X)
        if self.copy:
            X_ = X.copy()
        else:
            X_ = X
        for mat, indices in zip(self.transforms, self.indices_):
            X_[:, indices] = np.dot(X_[:, indices], mat)
        return X_


class CoordinateChangingDecisionTree(BaseEstimator):
    """
    Don't use this directly.
    """
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 max_features=None,
                 splitter='best',
                 presort=False,
                 criterion='gini',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 transform_dimension=None,
                 transform_repeats='auto',
                 usecols=None,
                 random_state=None):
        self.usecols = usecols
        self.splitter = splitter
        self.presort = presort
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = check_random_state(random_state)
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.transform_dimension = transform_dimension
        self.transform_repeats = transform_repeats
        self.initiate_tree_type()

    def initiate_tree_type(self):
        raise NotImplementedError('Base class must implement.')

    def fit(self, X, y, **kwargs):
        self.random_state = check_random_state(self.random_state)
        self.dt_ = self.tree_type_(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            criterion=self.criterion,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            splitter=self.splitter,
            presort=self.presort,
            max_leaf_nodes = self.max_leaf_nodes,
            min_impurity_split = self.min_impurity_split,
            random_state=self.random_state)

        self.transformer_ = RandomCoordinateChanger(
            transform_dimension=self.transform_dimension,
            transform_repeats=self.transform_repeats,
            usecols=self.usecols,
            random_state=self.random_state,
        )
        X_ = self.transformer_.fit_transform(X)
        self.dt_.fit(X_, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        X_ = self.transformer_.transform(X)
        return self.dt_.predict(X_, **kwargs)

    def predict_proba(self, X, **kwargs):
        X_ = self.transformer_.transform(X)
        return self.dt_.predict_proba(X_, **kwargs)

    @property
    def classes_(self):
        return self.dt_.classes_

    def _validate_X_predict(self, X, check_input):
        return self.dt_._validate_X_predict(X, check_input)


class CoordinateChangingDecisionTreeClassifier(
        CoordinateChangingDecisionTree,
        ClassifierMixin):
    """
    A decision tree classifier which performs a random change of
    coordinates before fitting or predicting.
    
    The parameters are mostly the same as sklearn's DecisionTreeClassifier, with
    the exception of transform_dimension, transform_repeats, and usecols, for
    which see RandomCoordinateChanger.

    It's probably pointless to use this directly: it's useful as part of an
    ensemble.
    """
    def initiate_tree_type(self):
        self.tree_type_ = tree.DecisionTreeClassifier


class CoordinateChangingDecisionTreeRegressor(
        CoordinateChangingDecisionTree):
    """
    Not ready for prime time.
    """
    def initiate_tree_type(self):
        self.tree_type_ = tree.DecisionTreeRegressor


class RandomCoordinateForestClassifier(ensemble.forest.ForestClassifier):
    """
    A random forest for which each tree performs a random change of coordinates
    before fitting or predicting.
    
    The parameters are largely the same as sklearn's RandomForestClassifier,
    with the exception of transform_dimension, transform_repeats, and usecols,
    for which see RandomCoordinateChanger.
    """
    def __init__(self,
                 transform_repeats='auto',
                 transform_dimension=None,
                 usecols=None,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomCoordinateForestClassifier, self).__init__(
            base_estimator=CoordinateChangingDecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes", "min_impurity_split",
                              "transform_repeats", "transform_dimension",
                              "usecols", "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
        self.transform_repeats = transform_repeats
        self.transform_dimension = transform_dimension
        self.usecols = usecols
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
