import pandas as pd
import numpy as np
import seaborn as sns
from xml.etree.ElementTree import fromstring
import matplotlib.pyplot as plt
from collections import defaultdict


figure_colors = sns.color_palette("BrBG").as_hex()
figure_colors_qualitative = sns.color_palette("colorblind", 6).as_hex()


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


def extract_scorers(row):
    goal = row["goal"]
    away_scorers = []
    home_scorers = []
    if goal is not np.nan:
        home_team_id = row["home_team_api_id"]
        root = fromstring(goal)
        # find all goals
        for value in root.findall("./value/[player1]"):
            # where the goal is not an own goal
            not_own_goal = value.find("stats/owngoals") is None
            if not_own_goal:
                # record the goal scorer and record if it was by the away or home team
                player = value.find("player1").text
                team = value.find("team").text
                if int(team) == home_team_id:
                    home_scorers.append(player)
                else:
                    away_scorers.append(player)
    return pd.Series(
        [",".join(home_scorers), ",".join(away_scorers)],
        index=["home_scorers", "away_scorers"],
    ).replace("", None)


def get_player_goal_counts(dataframe):
    # Create a defaultdict with a default value of 0
    player_goal_counts = defaultdict(int)
    player_teams = dict()

    # Iterate over each row in the DataFrame
    for _, row in dataframe.iterrows():
        player_ids_home = row["home_scorers"]
        player_ids_away = row["away_scorers"]
        team_home = row["home_team_api_id"]
        team_away = row["away_team_api_id"]

        # Handle home scorers
        if player_ids_home is not None:
            # Split the player IDs string into a list
            player_ids_list = player_ids_home.split(",")

            # Iterate over each player ID in the list
            for player_id_str in player_ids_list:
                player_id = int(player_id_str)
                player_goal_counts[player_id] += 1
                if player_teams.get(player_id) is None:
                    player_teams[player_id] = team_home
                elif player_teams[player_id] == -1:
                    continue
                elif player_teams[player_id] != team_home:
                    player_teams[player_id] = -1

        # Handle away scorers
        if player_ids_away is not None:
            player_ids_list = player_ids_away.split(",")
            for player_id_str in player_ids_list:
                player_id = int(player_id_str)
                player_goal_counts[player_id] += 1
                if player_teams.get(player_id) is None:
                    player_teams[player_id] = team_away
                elif player_teams[player_id] == -1:
                    continue
                elif player_teams[player_id] != team_away:
                    player_teams[player_id] = -1

    return player_goal_counts, player_teams


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


def transform_X(df, scaler):
    # scale df's numerical features
    numerical = df.select_dtypes(include=[np.number])
    numerical_scaled = scaler.transform(numerical)
    numerical_scaled_df = pd.DataFrame(
        numerical_scaled, columns=numerical.columns, index=numerical.index
    )

    # 'dummify' df's categorical features
    categorical = df.select_dtypes(include=[object])
    all_dummies = pd.DataFrame()
    for col in categorical.columns:
        column_dummies = pd.get_dummies(categorical[col], prefix=col, drop_first=True)
        all_dummies = pd.concat([all_dummies, column_dummies], axis=1)

    df_transformed = pd.concat([numerical_scaled_df, all_dummies], axis=1)
    return df_transformed


def paint_roc_figure(model, Y_test, X_test, pos_label, pos_position):
    fpr, tpr, thresholds = roc_curve(
        Y_test, model.predict_proba(X_test)[:, pos_position], pos_label=pos_label
    )

    roc_df = pd.DataFrame({"recall": tpr, "specificity": 1 - fpr})
    ax = roc_df.plot(
        x="specificity",
        y="recall",
        figsize=(5, 5),
        legend=False,
        color=figure_colors_qualitative[0],
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 0)
    ax.plot((1, 0), (0, 1), color=figure_colors_qualitative[1])
    ax.set_xlabel("specificity")
    ax.set_ylabel("recall")
    plt.title("Model ROC")
    plt.text(
        0.2,
        0.1,
        (
            "AUC:"
            f" {roc_auc_score(Y_test, model.predict_proba(X_test)[:,pos_position]):.2f}"
        ),
        fontsize=12,
    )
    sns.despine()


def paint_prc_figure(model, Y_test, X_test, pos_label, pos_position):
    precision, recall, thresholds = precision_recall_curve(
        Y_test, model.predict_proba(X_test)[:, pos_position], pos_label=pos_label
    )

    prc_df = pd.DataFrame({"recall": recall, "precision": precision})
    ax = prc_df.plot(
        y="precision",
        x="recall",
        figsize=(5, 5),
        legend=False,
        color=figure_colors_qualitative[0],
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_ylabel("precision")
    ax.set_xlabel("recall")
    plt.title("Model PRC")
    sns.despine()


def plot_roc_curve(tpr, fpr, scatter=True, ax=None):
    """
    Taken from
    https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    """
    if ax == None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x=fpr, y=tpr, ax=ax, color=figure_colors_qualitative[0])
    sns.lineplot(x=fpr, y=tpr, ax=ax, color=figure_colors_qualitative[0])
    sns.lineplot(x=[0, 1], y=[0, 1], color=figure_colors_qualitative[1], ax=ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


def plot_multiclass_roc(model, X_test, Y_test):
    """
    heavily based on
    https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
    """
    # Plots the Probability Distributions and the ROC Curves One vs Rest
    plt.figure(figsize=(12, 8))
    bins = [i / 20 for i in range(20)] + [1]
    classes = model.classes_
    y_proba = model.predict_proba(X_test)
    for i, c in enumerate(classes):
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = X_test.copy()
        df_aux["class"] = [1 if y == c else 0 for y in Y_test]
        df_aux["prob"] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 3, i + 1)
        sns.histplot(
            x="prob",
            data=df_aux,
            hue="class",
            color=figure_colors_qualitative,
            ax=ax,
            bins=bins,
        )
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 3, i + 4)
        fpr, tpr, thresholds = roc_curve(df_aux["class"], df_aux["prob"], pos_label=1)
        plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
        ax_bottom.set_title("ROC Curve")

        # Calculates the ROC AUC
        roc_auc = roc_auc_score(df_aux["class"], df_aux["prob"])
        plt.text(
            0.2,
            0.1,
            f"AUC: {roc_auc:.2f}",
            fontsize=12,
        )
    plt.tight_layout()
