.. _mean_median_imputer:

.. currentmodule:: feature_engine.imputation

MeanMedianImputer
=================

The MeanMedianImputer() is a univariate/single imputation algorithm for pandas DataFrames to replace missing data with either:

- The mean average of the data
- The median average of the data

And can be used over multiple columns at once without repeated method calls either by specifying a list of column names or letting the imputer automatically select all numerical variables in the data. Note: this imputation method only works on numerical data. The implementation mimics the standard scikit-learn pattern of learning the data first using .fit() and performing the imputation operation using .transform(). Data, with imputed values of the mean or median, is commonly used in data science and machine learning to train linear regression and logistic regression models. As this is a simple technique, other imputation techniques may be more suitable for complex data (discussed further below).

Choosing imputation type
------------------------

In a symmetric/normal distribution the mean and median are identical.

However, mean imputation can be sensitive to outliers when the underlying distribution is skewed. In this scenario the median can be a better representation of the average as the median is less prone to outliers.

Here is an example image showing this, taken from Wikipedia. The image links
to the use license.

.. figure::  ../../images/1024px-Relationship_between_mean_and_median_under_different_skewness.png
   :align:   center
   :target: https://commons.wikimedia.org/wiki/File:Relationship_between_mean_and_median_under_different_skewness.png

In this example:

- The positively skewed data has a left-shifted peak, but the mean is dragged up due to outliers in the long tail to the right
- In the negatively skewed data, there is a right-shifted peak but the mean is dragged down to due to outliers in the long tail to the left

It is clear the median is a better estimate of the central tendency of the skewed data.

Considerations when using MeanMedianImputer
-------------------------------------------

Both mean and median imputation are popular methods to handle missing numerical data because they are:

- Straight forward to understand
- Easy to implement
- Quick to apply to large datasets

Things to consider before using mean imputation or median imputation:

- Both methods assume the missing data is missing completely at random (MCAR). However, sometimes data is not missing at random due to relationships between other variables in the dataset which is not captured by these methods (MNAR - Missing Not Completely At Random). Hence these methods can introduce bias to the dataset. Other imputation methods might be more appropriate if the missing data is related to other other variables in the dataset, for example `sckit-learn's multiple imputation method IterativeImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`.
- Both methods can reduce the variance in a dataset since the replaced values represent the central tendency of the data.

Example usage
-------------

With the `fit()` method, the transformer learns and stores the mean or median values per
variable. Then it uses these values in the `transform()` method to transform the data.

Below a code example using the House Prices Dataset (more details about the dataset
:ref:`here <datasets>`).

Median imputation
~~~~~~~~~~~~~~~~~

First, let's load the data and separate it into train and test:

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.imputation import MeanMedianImputer

	# Load dataset
	data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
                                            data.drop(['Id', 'SalePrice'], axis=1),
                                            data['SalePrice'],
                                            test_size=0.3,
                                            random_state=0,
                                            )


Now we set up the :class:`MeanMedianImputer()` to impute in this case with the median
and only 2 variables from the dataset.

.. code:: python

	# set up the imputer
	median_imputer = MeanMedianImputer(
                           imputation_method='median',
                           variables=['LotFrontage', 'MasVnrArea']
                           )

	# fit the imputer
	median_imputer.fit(X_train)

With fit, the :class:`MeanMedianImputer()` learned the median values for the indicated variables and stored it in one of its attributes. We can observe the fitting parameters:

.. code:: python
        median_imputer.get_params()

returning a dict of our input parameters:

..code:: python
       {'imputation_method': 'median', 'variables': ['LotFrontage', 'MasVnrArea']}

And we can observe the learned median values:

.. code:: python
        print(median_imputer.imputer_dict_)

showing a dictionary where the keys are the column names and the values are the learned median values:

.. code:: python
       {'LotFrontage': 69.0, 'MasVnrArea': 0.0}


We can now go ahead and impute both
the train and the test sets.

.. code:: python

	# transform the data
	train_t= median_imputer.transform(X_train)
	test_t= median_imputer.transform(X_test)

Note that after the imputation, if the percentage of missing values is relatively big,
the variable distribution will differ from the original one (in red the imputed
variable):

.. code:: python

	fig = plt.figure()
	ax = fig.add_subplot(111)
	X_train['LotFrontage'].plot(kind='kde', ax=ax)
	train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
	lines, labels = ax.get_legend_handles_labels()
	ax.legend(lines, labels, loc='best')

.. image:: ../../images/medianimputation.png


Automatic detection of numeric columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If numeric columns aren't specified when instantiating a :class:`MeanMedianImputer()`, feature-engine will automatically select all of the numerical variables in the dataset. 
e.g. repeating the example above and excluding the `variables` argument in the imputer:

.. code:: python

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	from feature_engine.imputation import MeanMedianImputer

	# Load dataset
	data = pd.read_csv('houseprice.csv')

	# Separate into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
                                            data.drop(['Id', 'SalePrice'], axis=1),
                                            data['SalePrice'],
                                            test_size=0.3,
                                            random_state=0,
                                            )
        # set up the imputer
	median_imputer = MeanMedianImputer(
                           imputation_method='median',
                           )

We can print the chosen columns using the `imputer_dict_` attribute shown above and verify that all numeric columns have been selected by leveraging the pandas method `select_dtypes()`.
.. code:: python
        assert list(median_imputer.imputer_dict_.keys()) ==   list(X_train.select_dtypes(include='number').columns)



Additional resources
--------------------

In the following python Jupyter notebook you will find more details on the functionality of the
:class:`MeanMedianImputer()`, including how to select numerical variables automatically.
You will also see how to navigate the different attributes of the transformer to find the
mean or median values of the variables.

- `Jupyter notebook <https://nbviewer.org/github/feature-engine/feature-engine-examples/blob/main/imputation/MeanMedianImputer.ipynb>`_

For more details about this and other feature engineering methods check out these resources:


.. figure::  ../../images/feml.png
   :width: 300
   :figclass: align-center
   :align: left
   :target: https://www.trainindata.com/p/feature-engineering-for-machine-learning

   Feature Engineering for Machine Learning

|
|
|
|
|
|
|
|
|
|

Or read our book:

.. figure::  ../../images/cookbook.png
   :width: 200
   :figclass: align-center
   :align: left
   :target: https://packt.link/0ewSo

   Python Feature Engineering Cookbook

|
|
|
|
|
|
|
|
|
|
|
|
|

Both our book and course are suitable for beginners and more advanced data scientists
alike. By purchasing them you are supporting Sole, the main developer of Feature-engine.