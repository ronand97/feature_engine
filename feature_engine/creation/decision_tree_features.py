from itertools import combinations
from typing import List, Tuple, Union, Optional, Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _variables_numerical_docstring,
    _drop_original_docstring,
)

from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X_y,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    drop_original=_drop_original_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class DecisionTreeFeatures(BaseEstimator, TransformerMixin):
    """
    DecisionTreeFeatures() creates new variables by combining features using decision
    trees. DecisionTreeFeatures() will add as variables, the predictions of decision
    trees trained on 1, 2, 3 or more variables.

    DecisionTreeFeatures() is a useful method, when using linear models, because it
    captures non-linear relationships between features and the target by using trees.

    DecisionTreeFeatures() works only with numerical variables. Categorical variables
    need to be encoded into numbers first.

    Missing data should be imputed before using this transformer.

    More details in the :ref:`User Guide <dtrees_features>`.

    Parameters
    ----------
    {variables}

    strategy: integer, list or tuple, default=None
        Indicates how to combine the features in `variables` to train the decision
        trees. If None, then decision trees will be created with all possible feature
        combinations, from 1 to the total number of features. If integer, then all
        possible feature combinations from 1 to strategy will be used to train the
        trees. If list of integers, then all combinations of features of group size
        indicated in the list will be created. If tuple with tuples of variable names,
        then the decision trees will be created using the variables specified in the
        tuples.

        **Examples**


        The parameter `variables` takes ["var_A", "var_B", "var_C"]:

        - `strategy=1` returns 3 new features based on single variable decision trees.
        - `strategy=2` returns 6 new features based on single and double variable decision
        trees
        - `strategy=3` returns 7 new features based on single, double and triple variable
        decision trees.
        - `strategy=None` returns 7 new features based on single, double and triple
        variable decision trees.
        - `strategy=[1,2] returns 6 new features based on single and double variable
        decision trees
        - `stragety=(("var_A", "var_D"), ("var_B", "var_C")), returns 2 new features
        based on the combinations in the tuples.

        cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use cross_validate's default 5-fold cross validation

            - int, to specify the number of folds in a (Stratified)KFold,

            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)

            - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation.

    {drop_original}

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance of the tree. Comes from
        sklearn.metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    param_grid: dictionary, default=None
        The hyperparameters for the decision tree to test with a grid search. The
        `param_grid` can contain any of the permitted hyperparameters for Scikit-learn's
        DecisionTreeRegressor() or DecisionTreeClassifier(). If None, then param_grid
        will optimise the 'max_depth' over `[1, 2, 3, 4]`.

    regression: boolean, default = True
        Whether the decision tree is for regression or classification.

    random_state : int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.


    Attributes
    ----------
    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Fits the decision tree(s).

    transform:
        Adds new features.

    {fit_transform}

    See Also
    --------
    feature_engine.discretisation.DecisionTreeDiscretiser
    feature_engine.encoding.DecistionTreeEncoder
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.DecisionTreeRegressor

    References
    ----------
    .. [1] Niculescu-Mizil, et al. "Winning the KDD Cup Orange Challenge with Ensemble
        Selection". JMLR: Workshop and Conference Proceedings 7: 23-34. KDD 2009
        http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        strategy: Union[int, List[int], Tuple[tuple, ...]] = None,
        drop_original: bool = False,
        cv=3,
        scoring: str = "neg_mean_squared_error",
        param_grid: Optional[Dict[str, Union[str, int, float, List[int]]]] = None,
        regression: bool = True,
        random_state: Optional[int] = None,
    ) -> None:

        if (
                not isinstance(strategy, (int, list, tuple))
                and strategy is not None
        ):
            raise ValueError(
                f"strategy must integer, list or tuple. Got {strategy} instead."
            )
        # check user is not creating combinations of more variables
        # than the number of variables provided in the 'variables' param.
        if isinstance(strategy, int) and strategy > len(variables):
            raise ValueError(
                "strategy cannot be greater than the number of features to combine. "
                f"Got strategy={strategy} and {len(variables)} features."
            )

        if isinstance(strategy, list):
            # Check (1) user is not creating combinations of more variables
            # than the number of variables in the 'variables' param or
            # (2) the list is only comprised of integers.
            if (
                    max(strategy) > len(variables)
                    or not all(isinstance(feature, int) for feature in strategy)
            ):
                raise ValueError(
                    "strategy must be a list of integers and the maximum value cannot "
                    "be greater than the number of variables to combine. "
                )

        if isinstance(strategy, tuple) and variables is not None:

            # Check that the tuples only contains strings with variable names.
            if (
                not all(
                isinstance(element, (str, int)) and element in variables
                for tuples in strategy
                for element in tuples
                )
            ):
                raise ValueError(
                    "If strategy is tuples, the elements of the tuples must be strings"
                    "or integers present in the variable list entered in `variables`."
                )
        elif isinstance(strategy, tuple) and variables is None:
            raise ValueError(
                "if strategy is a tuple with variable names, `variables` cannot be "
                "None."
            )

        if not isinstance(drop_original, bool):
            raise TypeError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.output_features = strategy
        self.drop_original = drop_original
        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.variables = _check_input_parameter_variables(variables)
        self.param_grid = param_grid
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits decision trees with the provided the feature combinations.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just
            the variables to transform.

        y: pandas Series or np.array = [n_samples,]
            The target variable that is used to train the decision tree.
        """

        # only numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # basic checks
        X, y = check_X_y(X, y)
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # get all sets of variables that will be used to create new features
        self.variable_combinations_ = self._create_variable_combinations()
        self._estimators = {}

        # fit a decision tree for each set of variables
        for combo in self.variable_combinations_:
            self._estimators[combo] = self._make_decision_tree()
            self._estimators[combo].fit(X[combo], y)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add new features.

        Parameters
        ----------
        X: Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: Pandas dataframe.
            The original dataframe plus the additional features.
        """
        check_is_fitted(self)
        _check_X_matches_training_df(X, self.n_features_in_)

        # get new feature names for dataframe column names
        feature_names = self.get_feature_names_out(
            input_features=self.variable_combinations_
        )

        # create new features and add them to the original dataframe
        for (combo, estimator), name in zip(self._estimators, feature_names):
            X[name] = estimator.predict(X[combo])

        if self.drop_original:
            X.drop(columns=self.variables_, inplace=True)

        return X

    def _make_decision_tree(self):
        """Instantiate decision tree using grid search and cross-validation."""

        if self.param_grid:
            param_grid = self.param_grid
        else:
            param_grid = {"max_depth": [1, 2, 3, 4]}

        if self.regression is True:
            tree = DecisionTreeRegressor(random_state=self.random_state)
        else:
            tree = DecisionTreeClassifier(random_state=self.random_state)

        tree_model = GridSearchCV(
            tree, cv=self.cv, scoring=self.scoring, param_grid=param_grid
        )
        return tree_model

    def _create_variable_combinations(self):
        """
        Create a list of the different combinations of variables that will be used
        to create new features.
        """

        variable_combinations = []

        if isinstance(self.strategy, tuple):
            variable_combinations = [list(combo) for combo in self.strategy]

        elif isinstance(self.strategy, list):
            for num in self.strategy:
                variable_combinations += list(combinations(self.variables, num))

        else:
            if self.strategy is None:
                max_combo = len(self.variables)

            else:
                max_combo=self.strategy

            for num in range(1, max_combo):
                variable_combinations += list(combinations(self.variables, num))

        return variable_combinations

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            If input_features is None, then the names of all the variables in the
            transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the new features derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        feature_names = []

        for combo in self.variable_combinations_:
            if len(combo) == 1:
                feature_names.append(f"{combo[0]}_tree")

            else:
                combo_joined = "_".join(combo)
                feature_names.append(f"{combo_joined}_tree")

        if input_features is None:
            feature_names = self.feature_names_in_ + feature_names

        return feature_names

    def _get_unique_values_from_output_features(self) -> List:
        """
        Get unique values from output_features when it is a tuple.

        """
        unique_features = []

        # transform all elements in output_features to a list
        features_list = [
            list(features) if type(features) is tuple else [features]
            for features in self.output_features
        ]

        # merge all lists into 1 list
        for lst in features_list:
            unique_features += lst

        # get unique values
        unique_features = list(set(unique_features))

        return unique_features

    # TODO: Will most likely expand to include other checks
    def _check_output_features_are_permitted(self, unique_features: List = None) -> None:
        """
        Confirm that all elements of output_features are included in the dataset's
        numerical variables. If not raise an error.
        """
        # check if unique_features is a subset of self.variables_
        if not (set(unique_features).issubset(set(self.variables_))):
            raise ValueError(
                "All of the unique values of output_features are not a subset of "
                f"the dataset's variables. The unique features are {unique_features}. "
                f"The dataset's variables are {self.variables_}."
            )
