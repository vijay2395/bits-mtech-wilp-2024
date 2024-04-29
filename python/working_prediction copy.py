import pandas as pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
import warnings
from termcolor import colored
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from IPython.display import Image
from tabulate import tabulate


import warnings
warnings.filterwarnings("ignore")

#dataset description image
# display(Image(filename='/Users/vijaykumar/Desktop/PROJECT/final/REPORT/prediction_datasets/Cardiovascular_Disease_Dataset/dataset_description.png'))

#read the dataset
dataset = pandread_csv('/Users/vijaykumar/Desktop/PROJECT/final/REPORT/prediction_datasets/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv')
dataset.head().style.set_properties(**{'background-c olor':'blue','color':'white','border-color':'#8b8c8c'})

#display summary and statistics of dataset
dataset.describe().T.style.background_gradient(axis=0)

#null check
# dataset.isna().sum()


# age_range = f"Age Range: {dataset['age'].min()} - {dataset['age'].max()}"
# print(age_range)
# print("\n")

# gender_count = dataset['gender'].value_counts()
# print("Gender Count(female -> 0 , male -> 1):")
# print(gender_count)
# print("\n")

# chest_pain_counts = dataset['chestpain'].value_counts()
# print("Chest Paint Counts (0 -> Typical Angina, 1 -> Atypical Angina, 2 -> Non-Anginal Pain, 3 -> Asymptomatic):")
# print(chest_pain_counts)


# percentage_high_fasting_sugar = (dataset['fastingbloodsugar'].sum() / len(dataset)) * 100
# print(f"Percentage of patients with fasting blood sugar > 120 mg/dl: {percentage_high_fasting_sugar:.2f}%")

# average_max_heart_rate = dataset['maxheartrate'].mean()
# print(f"Average Maximum Heart Rate: {average_max_heart_rate:.2f}")

# average_resting_blood_pressure = dataset['restingBP'].mean()
# print(f"Average Blood Pressure (Resting) : {average_resting_blood_pressure:.2f} mm Hg")

# average_oldpeak = dataset['oldpeak'].mean()
# print(f"Average Oldpeak: {average_oldpeak:.2f}")

# vessels_range = f"No of Major Vessels Range: {dataset['noofmajorvessels'].min()} - {dataset['noofmajorvessels'].max()}"
# print(vessels_range)

# oldest_patient = dataset.loc[dataset['age'].idxmax()]
# print(f"Details of the Oldest Patient:\n{oldest_patient}")

# lowest_bp_patient = dataset.loc[dataset['restingBP'].idxmin()]
# print(f"Details of the Patient with the Lowest Resting Blood Pressure:\n{lowest_bp_patient}")

correlation_age_maxheartrate = dataset['age'].corr(dataset['maxheartrate'])
print(f"Correlation between Age and Maximum Heart Rate: {correlation_age_maxheartrate:.2f}")



# # Visualization: Heart Rate Max by Chest Pain
# plot.figure(figsize=(10, 5))
# sns.barplot(x='chestpain', y='maxheartrate', data=dataset, palette='viridis')
# plot.title(' Heart Rate Max by Chest Pain')
# plot.xlabel('Chest Pain Type')
# plot.ylabel('Max Heart Rate')
# plot.show()


# # Visualization: Heart Disease Presence by Chest Pain Type
# plot.figure(figsize=(10, 5))
# sns.countplot(x='chestpain', hue='target', data=dataset, palette='Set1')
# plot.title('Heart Disease Presence by Chest Pain Type')
# plot.xlabel('Chest Pain Type')
# plot.ylabel('Count')
# plot.legend(title='Heart Disease', labels=['No', 'Yes'])
# plot.show()

# # Visualization: Heart Disease Presence by Chest Pain Type
# plot.figure(figsize=(10, 5))
# sns.boxplot(x='target', y='serumcholestrol', data=dataset, palette='pastel')
# plot.title('Serum Cholesterol Distribution by Heart Disease Presence')
# plot.xlabel('Heart Disease Presence')
# plot.ylabel('Serum Cholesterol')
# plot.xticks(ticks=[0, 1], labels=['No Heart Disease', 'Heart Disease'])
# plot.show()


# # Visualization: Heart Rate by Gender and Heart Disease Presence
# plot.figure(figsize=(10, 5))
# sns.barplot(x='gender', y='maxheartrate', hue='target', data=dataset, palette='mako')
# plot.title(' Heart Rate by Gender and Heart Disease Presence')
# plot.xlabel('Gender')
# plot.ylabel('Max Heart Rate')
# plot.legend(title='Heart Disease', labels=['No', 'Yes'])
# plot.show()

# # Visualization: Age Distribution by Gender
# plot.figure(figsize=(10, 5))
# sns.histplot(x='age', hue='gender', data=dataset, palette='muted', multiple='stack', bins=15)
# plot.title('Age Distribution by Gender')
# plot.xlabel('Age')
# plot.ylabel('Count')
# plot.legend(title='Gender', labels=['Female', 'Male'])
# plot.show()

# # Visualization: Age vs. Max Heart Rate with Target Labels
# plot.figure(figsize=(10, 5))
# sns.scatterplot(x='age', y='maxheartrate', hue='target', data=dataset, palette='coolwarm', s=100)
# plot.title('Age vs. Max Heart Rate with Target Labels')
# plot.xlabel('Age')
# plot.ylabel('Max Heart Rate of Patients')
# plot.legend(title='Target', loc='upper right', labels=['No Heart Disease', 'Heart Disease'])
# plot.show()

# # Visualization: Serum Cholesterol Distribution
# plot.figure(figsize=(10, 5))
# sns.boxplot(x='serumcholestrol', data=dataset, palette='coolwarm')
# plot.title('Serum Cholesterol Distribution')
# plot.xlabel('Serum Cholesterol Distribution')
# plot.show()


# # Visualization: Slope Distribution of Peak Exercise ST Segment
# plot.figure(figsize=(10, 5))
# sns.countplot(x='slope', data=dataset, palette='viridis')
# plot.title('Slope Distribution of Peak Exercise ST Segment')
# plot.xlabel('Slope')
# plot.ylabel('Count')
# plot.show()


# target classes :
dataset.target.unique()

dataset = dataset.replace({'target' : {
                                    0 : 'Absence of Heart Disease',
                                    1 : 'Presence of Heart Disease',
        }}
)

dataset.head()

# Create X from DataFrame and y as Target
X_disease = dataset.drop(columns='target')
y = dataset.target

scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_disease)
X = pandDataFrame(scaler, columns=X_disease.columns)
X.describe().T.style.background_gradient(axis=0, cmap='icefire')


# Define a function to ploting Confusion matrix
def plot_confusion_matrix(y_test, y_prediction):
    '''Plotting Confusion Matrix'''
    cm = metrics.confusion_matrix(y_test, y_prediction)
    ax = plot.subplot()
    ax = sns.heatmap(cm, annot=True, fmt='', cmap="icefire")
    ax.set_xlabel('Predicted labels', fontsize=18)
    ax.set_ylabel('True labels', fontsize=18)
    ax.set_title('Confusion Matrix', fontsize=25)
    ax.xaxis.set_ticklabels(['Absence of Heart Disease','Presence of Heart Disease'])
    ax.yaxis.set_ticklabels(['Absence of Heart Disease','Presence of Heart Disease'])
    plot.show()


    # Define a function to ploting Classification report
def clfr_plot(y_test, y_pred) :
    ''' Plotting Classification report'''
    cr = pandDataFrame(metrics.classification_report(y_test, y_pred_rf, digits=3,
                                            output_dict=True)).T
    cr.drop(columns='support', inplace=True)
    sns.heatmap(cr, cmap='icefire', annot=True, linecolor='white', linewidths=0.5).xaxis.tick_top()

def clf_plot(y_pred) :
    '''
    1) Ploting Confusion Matrix
    2) Plotting Classification Report'''
    cm = metrics.confusion_matrix(y_test, y_pred)
    cr = pandDataFrame(metrics.classification_report(y_test, y_pred, digits=3, output_dict=True)).T
    cr.drop(columns='support', inplace=True)

    fig, ax = plot.subplots(1, 2, figsize=(15, 5))

    # Left: Confusion Matrix
    ax[0] = sns.heatmap(cm, annot=True, fmt='', cmap="icefire", ax=ax[0])
    ax[0].set_xlabel('Predicted labels', fontsize=18)
    ax[0].set_ylabel('True labels', fontsize=18)
    ax[0].set_title('Confusion Matrix', fontsize=25)
    ax[0].xaxis.set_ticklabels(['Absence of Heart Disease','Presence of Heart Disease'])
    ax[0].yaxis.set_ticklabels(['Absence of Heart Disease','Presence of Heart Disease'])

    # Right: Classification Report
    ax[1] = sns.heatmap(cr, cmap='icefire', annot=True, linecolor='white', linewidths=0.5, ax=ax[1])
    ax[1].xaxis.tick_top()
    ax[1].set_title('Classification Report', fontsize=25)
    plot.show()


dataset.target.value_counts()


# Splite Dataframe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Dictionary to define parameters to test in algorithm
parameters = {
    'n_estimators' : [50, 150, 500],
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'max_features' : ['sqrt', 'log2']
}

rf = RandomForestClassifier(n_jobs=-1)
rf_cv = GridSearchCV(estimator=rf, cv=20, param_grid=parameters).fit(X_train, y_train)

print('Tuned hyper parameters : ', rf_cv.best_params_)
print('accuracy : ', rf_cv.best_score_)



# Model :
rf = RandomForestClassifier(**rf_cv.best_params_).fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_score = round(rf.score(X_test, y_test), 3)
print('RandomForestClassifier score : ', rf_score)


y_test.value_counts()

# Ploting Confusion Matrix and Plotting Classification Report
clf_plot(y_pred_rf)




# Logistic Classifier

# Dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'class_weight' : ['balanced'],
    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}


lr = LogisticRegression()
lr_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10).fit(X_train, y_train)

print('Tuned hyper parameters : ', lr_cv.best_params_)
print('accuracy : ', lr_cv.best_score_)


lr = LogisticRegression(**lr_cv.best_params_).fit(X_train, y_train)


y_pred_lr = lr.predict(X_test)

lr_score = round(lr.score(X_test, y_test), 3)
print('LogisticRegression score : ', lr_score)


# Ploting Confusion Matrix and Plotting Classification Report
clf_plot(y_pred_lr)



#Support Vector Classifier

# Dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
}

svc = SVC()
svc_cv = GridSearchCV(estimator=svc, param_grid=parameters, cv=10).fit(X_train, y_train)

print('Tuned hyper parameters : ', svc_cv.best_params_)
print('accuracy : ', svc_cv.best_score_)

# Model
svc = SVC(**svc_cv.best_params_).fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

svc_score = round(svc.score(X_test, y_test), 3)
print('SVC Score : ', svc_score)

# Ploting Confusion Matrix and Plotting Classification Report
clf_plot(y_pred_svc)


# Decision Tree Classifier
# Dictionary to define parameters to test in algorithm
parameters = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'splitter' : ['best', 'random'],
    'max_depth' : list(np.arange(4, 30, 1))
        }

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator=tree, cv=10, param_grid=parameters).fit(X_train, y_train)


print('Tuned hyper parameters : ', tree_cv.best_params_)
print('accuracy : ', tree_cv.best_score_)

tree = DecisionTreeClassifier(**tree_cv.best_params_).fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

tree_score = round(tree.score(X_test, y_test), 3)
print('DecisionTreeClassifier Score : ', tree_score)


# Ploting Confusion Matrix and Plotting Classification Report
clf_plot(y_pred_tree)



#knn classifier

# Dictionary to define parameters to test in algorithm
parameters = {
    'n_neighbors' : list(np.arange(3, 50, 2)),
    'weights': ['uniform', 'distance'],
    'p' : [1, 2, 3, 4]
}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(estimator=knn, cv=10, param_grid=parameters).fit(X_train, y_train)

print('Tuned hyper parameters : ', knn_cv.best_params_)
print('accuracy : ', knn_cv.best_score_)

knn = KNeighborsClassifier(**knn_cv.best_params_).fit(X_train, y_train)

y_pred_knn = knn_cv.predict(X_test)

knn_score = round(knn.score(X_test, y_test), 3)
print('KNeighborsClassifier Score :', knn_score)

# Ploting Confusion Matrix and Plotting Classification Report
clf_plot(y_pred_knn)


#Gaussian Naive Bayes

gnb = GaussianNB().fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
gnb_score = round(gnb.score(X_test, y_test), 3)
print('GNB Score :', gnb_score)

clf_plot(y_pred_gnb)


#Algorithms Result
result = pandDataFrame({
    'Algorithm' : ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'GaussianNB'],
    'Score' : [rf_score, lr_score, svc_score, tree_score, knn_score, gnb_score]
})

result.sort_values(by='Score', inplace=True)

sns.set_palette("icefire")

fig, ax = plot.subplots(1, 1, figsize=(15, 5))

sns.barplot(x='Algorithm', y='Score', data=result)
ax.bar_label(ax.containers[0], fmt='%.3f')
ax.set_xticklabels(labels=result.Algorithm, rotation=300)
plot.show()


#final modeling based on result

rf = RandomForestClassifier(**rf_cv.best_params_)

rf.fit(X, y)
