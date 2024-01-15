## Ultimate 25k+ Matches Football Database

### Author: Tomas Balsevičius

To run the code, you will need to download the [data from Kaggle](https://www.kaggle.com/prajitdatta/ultimate-25k-matches-football-database-european)

Also, before running any code minor change to the database is needed - run database_edit.ipynb.

In database_exploration.ipynb, you will find the general overview of the database's tables.

In main_findings.ipynb you will find the main portion of the project.

To run the notebooks, first install dependencies using pip:

```Python
pip install -r requirements.txt
```

## Introduction

The dataset for this Sprint is the Ultimate 25k+ Matches Football Database. The main goal imposed for this task - explore the possibility to bet on Football matches. It is going to be a challenge to work with this large dataset, but our aim will be to find out what could make our bet successful. This project will consist of 3 major parts:

<ol>
    <li> EDA
    <ol>
        <li> Explore what leagues, teams and players score the most goals.
        <li> Analyse the time dynamics of football matches.
        <li> Explore the team attributes, trying to answer what makes a team perform better.
    </ol>
    <li> Statistical inference, dedicated to find out if there is such a thing as home advantage.
    <li> Modelling
    <ol>
        <li> Predict goals scored by the home or away team using:
        <ol>
            <li> Linear regression and one-hot encoding for the team id.
            <li> Linear regression and team attributes.
            <li> Logistic regression and team attributes
        </ol>
        <li> Predict match outcome using binomial and multinomial logistic regression.
        <li> Test the models in a betting game.
    </ol>
</ol>

## Technologies
<ul>
    <li>Python 3.11.1</li>
    <li>DuckDB (SQL interface)</li>
    <li>Pandas (dataframes),</li>
    <li>Seaborn and Matplotlib (visualisation),</li>
    <li>SciPY (statistical inference,)</li>
    <li>Statsmodels (explanatory modelling)</li>
    <li>Sklearn (modelling),</li>
</ul>

## Conclusions

We performed EDA using DuckDB to query the database, pandas to store and manipulate the data, seaborn and matplotlib - to visualise the data.

For statistical inference we utilised statsmodels and scipy. After hypothesis testing we are fairly certain that there is such a thing as home advantage.

We created some linear machine learning models (linear regression, logistic regression) to predict goals scored and the outcome of a football match. The best model for goals scored is just a bit better than random guessing, while the best model for match outcome prediction achieved 0.49 accuracy, with AUC 0.66 for home and away wins prediction. Ties are much harder to predict.

After trying to use the models for the betting game it is quite clear that betting on football matches is going to be risky. We didn't manage to get a positive income however hard we tried. Judging from these results, betting on football is not a good business model.

There are some things left to do and explore:
<ol>
    <li> Explore the player attributes, use them for modelling.
    <li> Don't use the latest team (or player) attributes for modelling. It should be more accurate to train with those attributes that are the most up-to-date at the time of the match.
    <li> total_score metric may be the best one we had in this project, but it could be improved. It could instead be score per league, since teams can migrate from a lower league to the top and vice-versa. For example, Bournemouth only played in 2016.
    <li> Use ordinal regression for goals scored, explore other models, not necessarily linear.
    
</ol>
Google Looker Studio dashboard can be found here: https://lookerstudio.google.com/reporting/51d1f099-c5bd-4dbf-96fd-f2db227abac2
