
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state
from sklearn import metrics, ensemble, tree, linear_model
from sklearn.model_selection import train_test_split

def random_point_on_sphere(dim=2, random_state=None):
    random_state = check_random_state(random_state)
    pt = 0
    norm = 0
    while norm < 1e-12:
        pt = random_state.normal(size=dim)
        norm = np.linalg.norm(pt)
    return pt / norm

def random_son(dim=2, random_state=None):
    random_state = check_random_state(random_state)
    result = -np.eye(dim, dtype=np.float)

    result[0, 0] = 1

    for i in range(2, dim+1):
        x = random_point_on_sphere(dim=i, random_state=random_state)
        result[:i, :i] = \
            result[:i, :i] - (np.dot(2*x, result[:i, :i])[:, np.newaxis]*x).T

    return result

class RandomCoordinateChanger(BaseEstimator, TransformerMixin):
    def __init__(self, transform_dimension=2, transform_repeats='auto', copy=True,
                 usecols=None, random_state=None):
        self.transform_dimension = transform_dimension
        self.transform_repeats = transform_repeats
        self.copy = copy
        self.usecols = usecols
        self.random_state = random_state

    def fit(self, X, y=None):
        self.random_state = check_random_state(self.random_state)
        X = np.array(X)
        m = X.shape[1]
        if self.transform_repeats == 'auto':
            self.real_repeats_ = m // self.transform_dimension
        else:
            self.real_repeats_ = self.transform_repeats

        def create_son():
            return random_son(dim=self.transform_dimension,
                              random_state=self.random_state)
        self.transforms = np.array([create_son()
                                    for i in range(self.real_repeats_)])

        if self.usecols is None:
            indices = np.array(list(range(m)))
        else:
            indices = np.array(self.usecols)
        def choose_indices():
            return self.random_state.choice(indices,
                                            size=self.transform_dimension,
                                            replace=False)
        self.indices = np.array([choose_indices()
                                 for i in range(self.real_repeats_)])
        return self

    def transform(self, X):
        if self.copy:
            X_ = X.copy()
        else:
            X_ = X
        for mat, indices in zip(self.transforms, self.indices):
            X_[:, indices] = np.dot(X_[:, indices], mat)
        return X_

class CoordinateChangingDecisionTree(BaseEstimator):
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
                 transform_dimension=2,
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
        CoordinateChangingDecisionTree):
    def initiate_tree_type(self):
        self.tree_type_ = tree.DecisionTreeClassifier

class CoordinateChangingDecisionTreeRegressor(
        CoordinateChangingDecisionTree):
    def initiate_tree_type(self):
        self.tree_type_ = tree.DecisionTreeRegressor

class RandomCoordinateForestClassifier(ensemble.forest.ForestClassifier):
    def __init__(self,
                 transform_repeats='auto',
                 transform_dimension=2,
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

if __name__ == '__main__':

    random_state = np.random.RandomState(72)
    from sklearn.datasets import make_classification, make_regression, make_friedman2

    # X, y = make_regression(n_samples=5000,
    #                        noise=0.0,
    #                        n_features=2,
    #                        n_informative=2,
    #                        random_state=random_state)

    # # X, y = make_friedman2(n_samples=1200,
    # #                       noise=0.0,
    # #                       random_state=random_state)

    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=1000,
    #                                                     random_state=random_state)

    # def check_mae(est):
    #     est.fit(X_train, y_train)
    #     y_pred = est.predict(X_test)
    #     print(metrics.mean_absolute_error(y_test, y_pred))

    # transform_dimension=2
    # transform_repeats=1

    # tre = tree.DecisionTreeRegressor(
    #     random_state=random_state,
    #     max_features=2,
    #     min_samples_split=1,
    #     max_depth=None)
    # bag_tre = ensemble.BaggingRegressor(
    #     max_features=1.0,
    #     n_estimators=300,
    #     base_estimator=tre,
    #     random_state=random_state)
    # print('plain bag')
    # check_mae(bag_tre)

    # clf = CoordinateChangingDecisionTreeRegressor(
    #     criterion='mae',
    #     random_state=random_state,
    #     max_features=2,
    #     transform_dimension=transform_dimension,
    #     transform_repeats=transform_repeats,
    #     min_samples_split=1,
    #     max_depth=None)
    # bag = ensemble.BaggingRegressor(
    #     max_features=1.0,
    #     n_estimators=300,
    #     base_estimator=clf,
    #     random_state=random_state)

    # print('my bag')
    # check_mae(bag)

    # est = linear_model.LinearRegression()
    # print('linear')
    # check_mae(est)

    # est = ensemble.RandomForestRegressor(n_estimators=600)
    # print('rf')
    # check_mae(est)


    # exit()

    X, y = make_classification(n_samples=10000,
                               n_classes=5,
                               n_features=20,
                               n_informative=17,
                               n_redundant=2,
                               n_repeated=0,
                               n_clusters_per_class=2,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=random_state,
                                                        stratify=y)

    def check_accuracy(est):
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        print(metrics.accuracy_score(y_test, y_pred))

    transform_dimension=4
    transform_repeats=20

    coordinate_rf = RandomCoordinateForestClassifier(
        random_state=random_state,
        max_features=7,
        n_estimators=300)
    print('my rf')
    check_accuracy(coordinate_rf)
        
    tre = tree.DecisionTreeClassifier(
        random_state=random_state,
        max_features=10,
        min_samples_split=1,
        max_depth=None)
    bag_tre = ensemble.BaggingClassifier(
        max_features=1.0,
        n_estimators=300,
        base_estimator=tre,
        random_state=random_state)
    print('plain bag')
    check_accuracy(bag_tre)


    clf = CoordinateChangingDecisionTreeClassifier(
        random_state=random_state,
        max_features=10,
        transform_dimension=transform_dimension,
        transform_repeats=transform_repeats,
        min_samples_split=1,
        max_depth=None)
    bag = ensemble.BaggingClassifier(
        max_features=1.0,
        n_estimators=300,
        base_estimator=clf,
        random_state=random_state)
    print('my bag')
    check_accuracy(bag)

    rf = ensemble.RandomForestClassifier(
        random_state=random_state,
        max_features=10,
        n_estimators=300)
    print('rf')
    check_accuracy(rf)