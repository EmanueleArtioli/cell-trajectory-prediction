import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ESPLORATORY ANALYSIS

df = pd.read_csv("Trajectories.csv", delimiter=",")
# some attributes are outside scope, dropped
df = df.iloc[:, :4]
df.columns = ['id', 'frame', 'x', 'y']
print(df)
# visualizing
plt.figure(figsize=(20, 15))
# showing everything is messy, we concentrate on the first 200 ids
plt.scatter(df.loc[df['id'] < 200, 'x'], df.loc[df['id'] < 200, 'y'], s=25, c=df.loc[df['id'] < 200, 'id'])

# to speed up debugging, we use the first 200 ids temporarily, comment bottom line for full results.
# df = df.loc[df['id'] <= 200]

# STITCHING
                
# We look for pairs of ids where one follows the other (the former stops appearing before
# the latter starts, and where the closest frames are close in position and movement speed)
def stitching(data, max_time_distance = 5, tolerance = 0.3):
    data_stitched = data
    # speed calculated as the euclidean distance between the mean squared errors of x and y
    sem = data_stitched[['id', 'x', 'y']].groupby('id').sem()
    sem.columns = ['speed_x', 'speed_y']
    # first and last appearances of each id with speed
    max_frames = data_stitched[['id', 'frame']].groupby('id').max().reset_index().merge(data_stitched).merge(sem, on='id')
    min_frames = data_stitched[['id', 'frame']].groupby('id').min().reset_index().merge(data_stitched).merge(sem, on='id')
    # stitching:
    for i in max_frames.id:
        # among the first appearances we only check on those close enough in time (for efficiency)
        # for the same reason, we save i's row
        current_max = max_frames.loc[max_frames['id'] == i, ].reset_index()
        lower_x_bound = current_max.x - tolerance * 1000
        upper_x_bound = current_max.x + tolerance * 1000
        lower_y_bound = current_max.y - tolerance * 1000
        upper_y_bound = current_max.y + tolerance * 1000
        lower_speed_x_bound = current_max.speed_x - tolerance * current_max.speed_x
        upper_speed_x_bound = current_max.speed_x + tolerance * current_max.speed_x
        lower_speed_y_bound = current_max.speed_y - tolerance * current_max.speed_y
        upper_speed_y_bound = current_max.speed_y + tolerance * current_max.speed_y
        close_mins = min_frames.loc[min_frames['frame'] - max_time_distance - current_max.frame.values < 0]
        close_mins = close_mins.loc[close_mins['frame'] + max_time_distance - current_max.frame.values > 0]
        for j in close_mins.id:
            current_min = close_mins.loc[close_mins['id'] == j, ].reset_index()
            conditions = [lower_x_bound < current_min.x, upper_x_bound > current_min.x, lower_y_bound < current_min.x, upper_y_bound > current_min.y, lower_speed_x_bound < current_min.speed_x, upper_speed_x_bound > current_min.speed_x, lower_speed_y_bound < current_min.speed_y, upper_speed_y_bound > current_min.speed_y ]
            conditions = [item.bool() for item in conditions]
            if all(conditions):
                data_stitched.loc[data_stitched['id'] == int(current_min.id), 'id'] = int(max_frames.loc[max_frames['id'] == int(current_max.id), ].id)
    return data_stitched

df = stitching(df)

# check the reduction in ids
df['id'].nunique()

# check visually
plt.figure(figsize=(20, 15))
plt.scatter(df.loc[df['id'] < 200, 'x'], df.loc[df['id'] < 200, 'y'], s=25, c=df.loc[df['id'] < 200, 'id'])
# as we can see, many more trajectories are present in the first 200 ids, and their patterns look all smooth.

# PREDICTIONS

listed_x = df.groupby('id')['x'].apply(list)
listed_y = df.groupby('id')['y'].apply(list)
# Predicting trajectories from too few points is useless, we therefore set a threshold at 30.
to_drop = []
for i in listed_y.index:
    if len(listed_y.loc[i]) < 30:
        to_drop.append(i)
listed_y.drop(to_drop, inplace=True)
listed_x.drop(to_drop, inplace=True)

# ORDINARY LEAST SQUARES

# Since we need to predict x and y using the same attributes, regression seems the right approach, we begin with ols.
# Molecules move independently from each other, therefore every id needs a dedicated train and test set.
ols_scores = []
y_pred = []
plt.figure(figsize=(20, 15))
for x, y in zip(listed_x, listed_y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    reg = LinearRegression().fit(x_train, y_train)
    y_pred.append(reg.predict(x_test))
    ols_scores.append(reg.score(x_test, y_test))
    # visualizing
    # real ones
    plt.scatter(x_test, y_test, s=25, c='blue')
    # predicted ones
    plt.scatter(x_test, reg.predict(x_test), s=25, c='red')
plt.show()
print('OLS average score is', sum(ols_scores)/len(ols_scores))

# It's clear how, although the trajectories are somewhat followed, the linear approximation in unsatisfying.
    
# POLYNOMIAL REGRESSION

# Most trajectories aren't linear, therefore a polynomial regression has a good chance of better fitting our data
degree=4
ols_scores = []
y_pred = []
plt.figure(figsize=(20, 15))
for x, y in zip(listed_x, listed_y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(x_train, y_train)
    y_pred.append(polyreg.predict(x_test))
    ols_scores.append(polyreg.score(x_test, y_test))
    # visualizing
    # real ones
    plt.scatter(x_test, y_test, s=25, c='blue')
    # predicted ones
    plt.scatter(x_test, polyreg.predict(x_test), s=25, c='red')
plt.show()
print('PolyReg average score is', sum(ols_scores)/len(ols_scores), 'with 4 degrees of freedom')

# As foreseen, a non-linear model is much better at predicting the trajectories.