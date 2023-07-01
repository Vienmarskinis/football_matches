import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SequentialFeatureSelector
from itertools import product
# from sklearn.metrics import f1_score

figure_colors = sns.color_palette("BrBG").as_hex()


def analyse_column(df, col_name: str, fig_title: str, *args, **kwargs) -> None:
    """Creates countplot figure with title, despined and labeled.

    Parameters
    ----------
    df : pd.Dataframe
        pandas dataframe containing the column col_name
    col_name : str
        The feature that will be analysed.
    fig_title : str
        The title of the figure.
    """
    sns.countplot(x=df[col_name], *args, **kwargs)
    plt.title(fig_title)
    sns.despine(bottom=True)
    add_value_labels(plt.gca())


def add_value_labels(ax, spacing=1, units="") -> None:
    """Add labels to the end of each bar in a bar chart.

    Taken from:
    https://stackoverflow.com/questions/28931224/how-to-add-value-labels-on-a-bar-chart

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib object containing the axes of the plot to annotate.
    spacing : int
        The distance between the labels and the bars.
    units : str
        The symbol(s) that will be appended to the end of the labels.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = "bottom"

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = "top"

        # Use Y value as label and format number with one decimal place
        label = f"{y_value:.0f}{units}"

        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha="center",  # Horizontally center label
            va=va,
        )  # Vertically align label differently for
        # positive and negative values.


def percent_plot(data: pd.DataFrame, x: str, hue: str, title: str, **kwargs) -> None:
    """Creates a barplot displaying the proportions of categories in a group in percentages.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame containing x and hue columns.
    x : str
        The name of the column used to split the plot into several groups.
    hue : str
        The name of the column used to split groups into several categories.
    title : str
        The name of the figure.
    """
    hue_percent = f"{hue}_percent"
    df_counts = data.groupby(x)[hue].value_counts(normalize=True)
    df_counts = df_counts.mul(100).rename(hue_percent).reset_index()
    sns.barplot(
        x=df_counts[x],
        y=df_counts[hue_percent],
        hue=df_counts[hue],
        palette=figure_colors,
        **kwargs,
    )
    plt.title(title)
    add_value_labels(plt.gca(), units="%")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, title=hue)
    plt.gcf().set_size_inches(8, 6)
    sns.despine(bottom=True)


def joint_scatterplot_with_title(title, *args, **kwargs):
    """Creates a jointplot with proper title. See sns.jointplot for available arguments.

    Parameters
    ----------
    title : str
        The name of the figure.
    """
    joint_grid = sns.jointplot(kind="scatter", alpha=0.3, height=6, *args, **kwargs)
    joint_grid.fig.subplots_adjust(top=0.95)
    plt.suptitle(title)


def correlation_bar(df, feature, title, kind="pearson"):
    """Creates a correlation bar for the specified feature.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing the feature.
    feature : str
        The name of the feature for which the correlations will be calculated.
    title : str
        The name of the figure.
    kind : str
        The kind of correlation, check pandas .corr() method for available kinds.
    """
    plt.figure(figsize=(5, 4))
    heatmap = sns.heatmap(
        df.corr(kind)[[feature]].sort_values(by=feature, ascending=False).drop(feature),
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="BrBG",
    )
    heatmap.set_title(title)


def validate_model(Fold_validator, model_input, X, Y):
    """Validate the model using Kfold validation.

    Parameters
    ----------
    Fold_validator
        sklearn KFold object.
    model_input
        sklearn classifier object. Tested to work with logistic regression.
    X : pd.Dataframe
        The raw input matrix.
    Y : pd.Dataframe
        The true classes to be used for classification.
    Returns
    -------
    mean intercept
    mean coefficients
    true Y vector
    predicted Y vector
    """
    scaler = StandardScaler()
    model = model_input
    Y_predictions = []
    Y_truths = []
    intercept = []
    coefs = []
    for train_index, test_index in Fold_validator.split(X, Y):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        Y_train = Y.iloc[train_index]
        Y_test = Y.iloc[test_index]
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model = model.fit(X_train, Y_train)
        Y_predictions.extend(model.predict(X_test))
        Y_truths.extend(Y_test)
        coefs.append(model.coef_)
        intercept.append(model.intercept_)
    return (
        np.array(intercept).mean(),
        np.array(coefs).mean(axis=0),
        Y_truths,
        Y_predictions,
    )


def tune_log_reg(KFold, X, Y):
    """Tune logistic regression parameters.

    TODO: add "combs" variable as input

    Parameters
    ----------
    Fold_validator
        sklearn KFold object.
    X : pd.Dataframe
        The raw input matrix.
    Y : pd.Dataframe
        The true classes to be used for classification.
    Returns
    -------
    best F1 score
    best parameters
    """
    f1_list = []
    combs = list(
        product(
            np.arange(1, 4, 0.1),
            ["l1", "l2"],
            [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
        )
    )
    for comb in combs:
        weight_1, penalty, C = comb
        model = LogisticRegression(
            solver="liblinear", class_weight={0: 1, 1: weight_1}, penalty=penalty, C=C
        )
        try:
            _, _, Y_truths, Y_predictions = validate_model(KFold, model, X, Y)
            if sum(Y_predictions) > 0:
                f1 = f1_score(Y_truths, Y_predictions)
                f1_list.append(f1)
            else:
                f1_list.append(0)
        except:
            f1_list = f1_list.append(0)
    best_f1 = np.max(f1_list)
    best_f1_idx = np.argmax(f1_list)
    best_params = combs[best_f1_idx]
    return best_f1, best_params


def plot_decision_boundary(title, model, X, Y, coefs_to_use):
    plt.figure()
    # Retrieve the model parameters.
    b = model.intercept_[0]
    w1, w2 = model.coef_.T[coefs_to_use]
    # Calculate the intercept and gradient of the decision boundary.
    c = -b / w2
    m = -w1 / w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, "k", lw=1, ls="--")
    plt.fill_between(xd, yd, ymin, color="tab:orange", alpha=0.2)
    plt.fill_between(xd, yd, ymax, color="tab:blue", alpha=0.2)

    negatives = X[Y == 0].T
    positives = X[Y == 1].T

    plt.scatter(*positives, s=8, alpha=0.5)
    plt.scatter(*negatives, s=8, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
