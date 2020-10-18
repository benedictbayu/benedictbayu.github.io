---
layout: post
title:  "FIFA 19 Data Analysis"
date:   2020-10-15 19:27 +0700
categories: jekyll update
---
# FIFA 19 Data Analysis Using Sckit Learn and Keras

### Introduction

![2yPHMB.jpg](https://iili.io/2yPHMB.jpg)

FIFA 19 is a football simulation game which developed by Electronics Arts. This game was introduced
on June 6, 2018 at the E3 2018 event. This game was released on September 28, 2018 for PlayStation 3,
PlayStation 4, Xbox One, Nintendo Switch, and Microsoft Windows consoloes. This game features the UEFA Club Competitions
including UEFA Champions League and UEFA Europa League for the very first time. FIFA 19 is also the last FIFA game series 
released for the PlayStation 3 console.

In this chance, i will do the data analysis for FIFA 19 dataset. This dataset was directly scrapped from [Sofifa][sofifa], a website that contains complete data about players and teams in FIFA 19.

### Data Cleaning

You can see the data used for the analysis below. 

![2bGkpR.jpg](https://iili.io/2bGkpR.jpg)

![2bGekv.jpg](https://iili.io/2bGekv.jpg)

![2bGOYJ.jpg](https://iili.io/2bGOYJ.jpg)

![2bGN2a.jpg](https://iili.io/2bGN2a.jpg)

![2bG8Ip.jpg](https://iili.io/2bG8Ip.jpg)

![2bGShN.jpg](https://iili.io/2bGShN.jpg)

This dataset consists of 18207 rows and 89 columns. However, i won't use all 89 columns, i will drop and summarize some of the columns.

This dataset will be uploaded into Google Colab using `pd.read_csv` code which will read the csv file into dataframe and 
will be named `fifa19`

Then i will check the missing values in the dataset using `fifa19.isnull().sum().sum()` code. There are 77000 missing values approximately in this dataset.

![26iEVR.jpg](https://iili.io/26iEVR.jpg)

Next, we will check which columns that contain the missing values ​​with `fifa19.isnull().sum().sort_values​​()` code. Here is a list of the columns with missing values.

![26PL6N.jpg](https://iili.io/26PL6N.jpg)   ![26PsGp.jpg](https://iili.io/26PsGp.jpg) 

![26sqfp.jpg](https://iili.io/26sqfp.jpg)   ![26sKsR.jpg](https://iili.io/26sKsR.jpg)
 
We will fill the missing values. The filling methods are varies, some are filled with mean, some with the mode value, some will only be filled in with 0 value. I fill these missing values using one of pandas library functions with `fifa19.fillna()` code.

{% highlight ruby %}
fifa19['ShortPassing'].fillna(fifa19['ShortPassing'].mean(), inplace=True)
fifa19['Volleys'].fillna(fifa19['Volleys'].mean(), inplace=True)
fifa19['Dribbling'].fillna(fifa19['Dribbling'].mean(), inplace=True)
fifa19['Curve'].fillna(fifa19['Curve'].mean(), inplace=True)
fifa19['FKAccuracy'].fillna(fifa19['FKAccuracy'].mean(), inplace=True)
fifa19['LongPassing'].fillna(fifa19['LongPassing'].mean(), inplace=True)
fifa19['BallControl'].fillna(fifa19['BallControl'].mean(), inplace=True)
fifa19['HeadingAccuracy'].fillna(fifa19['HeadingAccuracy'].mean(), inplace=True)
fifa19['Finishing'].fillna(fifa19['Finishing'].mean(), inplace=True)
fifa19['Crossing'].fillna(fifa19['Crossing'].mean(), inplace=True)
fifa19['Weight'].fillna('200 lbs', inplace=True)
fifa19['Contract Valid Until'].fillna(2019, inplace=True)
fifa19['Height'].fillna("5'11", inplace=True)
fifa19['Loaned From'].fillna('None', inplace=True)
fifa19['Joined'].fillna('Jul 1, 2018', inplace=True)
fifa19['Jersey Number'].fillna(8, inplace=True)
fifa19['Body Type'].fillna('Normal', inplace=True)
fifa19['Position'].fillna('ST', inplace=True)
fifa19['Club'].fillna('No Club', inplace=True)
fifa19['Work Rate'].fillna('Medium/ Medium', inplace=True)
fifa19['Skill Moves'].fillna(fifa19['Skill Moves'].median(), inplace=True)
fifa19['Weak Foot'].fillna(3, inplace=True)
fifa19['Preferred Foot'].fillna('Right', inplace=True)
fifa19['International Reputation'].fillna(1, inplace=True)
fifa19['Wage'].fillna('€200K', inplace=True)
fifa19['Release Clause'].fillna('0', inplace=True)

fifa19.fillna(0, inplace=True)
{% endhighlight %}

We will check the missing values once more just to make sure that we have removed the missing values using `fifa19.isnull().sum().sum()` code.

![26ilDJ.jpg](https://iili.io/26ilDJ.jpg)

We can see that now there is 0 missing value in the dataset. We have completely removed the missing values in the dataset. 

Then we will drop some features like `'Unnamed: 0'`, `'Photo'`, `'Club Logo'`, `'Flag'` with `fifa19.drop(['Unnamed: 0', 'Photo', 'Club Logo', 'Flag'], axis=1, inplace=True)` because it doesn't provide any information regarding the target we want to predict, which is `Overall`.

Next, we will summarize some of the features in this dataset by making 6 new features that contains the average values of the features we have summarized

{% highlight ruby %}
def pace(fifa):
  return int(round((fifa[['SprintSpeed', 'Acceleration']].mean()).mean()))

def passing(fifa):
  return int(round((fifa[['ShortPassing', 'Crossing', 'Vision', 'LongPassing', 'Curve', 'FKAccuracy']].mean()).mean()))

def defending(fifa):
  return int(round((fifa[['StandingTackle', 'Marking', 'Interceptions', 'HeadingAccuracy', 'SlidingTackle']].mean()).mean()))

def shooting(fifa):
  return int(round((fifa[['Finishing', 'LongShots', 'ShotPower', 'Volleys', 'Positioning', 'Penalties']].mean()).mean()))

def dribbling(fifa):
  return int(round((fifa[['Dribbling', 'BallControl', 'Agility', 'Balance']].mean()).mean()))

def physical(fifa):
  return int(round((fifa[['Strength', 'Stamina', 'Aggression', 'Jumping']].mean()).mean()))
{% endhighlight %}

And then we will add this 6 new features into dataframe and the features that were used to create these 6 new features will be dropped.

{% highlight ruby %}
fifa19['Pace'] = fifa19.apply(pace, axis=1)
fifa19['Passing'] = fifa19.apply(passing, axis=1)
fifa19['Defending'] = fifa19.apply(defending, axis=1)
fifa19['Shooting'] = fifa19.apply(shooting, axis=1)
fifa19['Dribbling'] = fifa19.apply(dribbling, axis=1)
fifa19['Physical'] = fifa19.apply(physical, axis=1)

fifa19.drop(['SprintSpeed', 'Acceleration', 'ShortPassing', 'Crossing', 'Vision', 'LongPassing', 'Curve', 'FKAccuracy', 'StandingTackle', 'Marking', 'Interceptions', 
             'HeadingAccuracy', 'SlidingTackle', 'Finishing', 'LongShots', 'ShotPower', 'Volleys', 'Positioning', 'Penalties', 'BallControl', 'Agility', 'Balance', 
             'Strength', 'Stamina', 'Aggression', 'Jumping'], axis=1, inplace=True)
{% endhighlight %}

Next, we will check the dimension from the dataset using `fifa19.shape` code

![26suzG.jpg](https://iili.io/26suzG.jpg)

It can be seen that previously we had 89 columns, but now there are 64 column because we have dropped some of the features and summarized them. 

Next, we will do the data cleaning for `weight`, `value`, `wage`, and `release clause` features. 
For `weight`, we will remove the **lbs** so we can convert the data type from *object* into *float*.

{% highlight ruby %}
def weight_cleaning(weight):
  out = weight.replace('lbs', '')
  return float(out)

fifa19['Weight'] = fifa19['Weight'].apply(lambda x : weight_cleaning(x))
{% endhighlight %}

For `value`, `wage`, and `release clause` we will remove the **€**, **M**, and **K** so we can convert the data type from *object* into *float*.

{% highlight ruby %}
def wage_cleaning(wage):
  out = wage.replace('€', '')
  if 'M' in out:
    out = float(out.replace('M', ''))*1000000
  elif 'K' in out:
    out = float(out.replace('K', ''))*1000
  return float(out)

fifa19['Value'] = fifa19['Value'].apply(lambda x : wage_cleaning(x))
fifa19['Wage'] = fifa19['Wage'].apply(lambda x : wage_cleaning(x))
fifa19['Release Clause'] = fifa19['Release Clause'].apply(lambda x : wage_cleaning(x))
{% endhighlight %}

### Data Exploration

Now we will see the insights we can get from the dataset.
First we will create a function the see the list of the players from specific country. I will use players from Italy as an example.

{% highlight ruby %}
def country(x):
  return fifa19[fifa19['Nationality'] == x][['Name', 'Overall', 'Potential', 'Position']]

country('Italy')
{% endhighlight %}

Next, we will create a function to show the squad from specific club in dataset. I will use Milan as an example.

{% highlight ruby %}
def club(x):
  return fifa19[fifa19['Club'] == x][['Name', 'Jersey Number', 'Position', 'Overall', 'Nationality', 'Age', 'Wage', 'Value', 'Contract Valid Until']]

club('Milan')
{% endhighlight %}

Next, we will make visualizations to get more insights from the dataset. 

We will see the **Preferred Foot** from footballers in FIFA 19 game. From the graph below, we can see that most footballers in FIFA 19 are right footed.

![266vmG.jpg](https://iili.io/266vmG.jpg)

We will see the **International Reputation** from footballers in FIFA 19. From the chart below, most footballers in FIFA 19 have no or less experiences in international matches, as we can see that most of them have value 1 for the International Reputation. The higher the value, it shows that the footballer already has good experience in international matches and the easier it is for the footballer to be targeted by big clubs.

![266L7e.jpg](https://iili.io/266L7e.jpg)

We will see the **Weak Foot** from footballer in FIFA 19. From the chart below, most footballers in FIFA 19 have a decent ability to shoot by their weak foot, as we can see that most of them has score 3 for the Weak Foot. the higher the value, it shows that the footballers have excellent ability to shoot the ball by their weak foot and vice versa.

![26PMve.jpg](https://iili.io/26PMve.jpg)

We will see the footballers distribution based on their **Position**. From the graph below, it can be seen that most footballers in FIFA 19 are playing as Striker (ST)

![26PE37.jpg](https://iili.io/26PE37.jpg)

We will see the footballer **Wage** in FIFA 19. From the plot below, it can be seen that most footballers in FIFA 19 have wage below €260K per week and only few footballers who have wage above €260K per week.

![26PlG2.jpg](https://iili.io/26PlG2.jpg)

We will see the footballer **Work Rate** in FIFA 19. From the graph below, it can be seen that most footballers have Medium/Medium work rates which mean that they have decent effort to help the attack and defense. As a comparasion, footballers that have High/Low work rates put great effort in helping the attack but bad at helping the defense.

![26P04S.jpg](https://iili.io/26P04S.jpg)

We will see **Top 25 Countries with most footballers** in FIFA 19. From the graph below, it can be seen that most footballers in FIFA 19 come from England.

![26PGa9.jpg](https://iili.io/26PGa9.jpg)

We will see the **Overall** from the countries in FIFA 19. From the graph below it can be seen that footballers from Brazil and Spain have the highest overall among others.

![26Phjj.jpg](https://iili.io/26Phjj.jpg)

We will see the **Wage** distribution for footballers in FIFA 19 based on their **nationality**. We will take top 10 countries with most footballers in FIFA 19. From the graph below, it can be seen footballers from Brazil has the highest average wages among others.

![26PXTb.jpg](https://iili.io/26PXTb.jpg)

We will see the **Wage** distribution based on their **club**. We will take top 10 clubs with the highest wage in FIFA 19. From the graph below, it can be seen that Real Madrid is the club with the highest average wages among others.

![26PVyu.jpg](https://iili.io/26PVyu.jpg)

We will see **best footballers in each position based on their *Potential* and *Overall* scores**

{% highlight ruby %}
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Potential'].idxmax()][['Position', 'Name', 'Club', 'Age', 'Nationality', 'Potential']] #Untuk dilihat berdasarkan overall, cukup ganti kolom 'Potential' menjadi 'Overall'
{% endhighlight %}

![26t4wX.jpg](https://iili.io/26t4wX.jpg)

![26t6tn.jpg](https://iili.io/26t6tn.jpg)

We will see **worst footballers in each position based on their *Potential* and *Overall* scores**

{% highlight ruby %}
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Potential'].idxmin()][['Position', 'Name', 'Club', 'Age', 'Nationality', 'Potential']] #Untuk dilihat berdasarkan overall, cukup ganti kolom 'Potential' menjadi 'Overall'
{% endhighlight %}

![26tLPf.jpg](https://iili.io/26tLPf.jpg)

![26tsMG.jpg](https://iili.io/26tsMG.jpg)

We will see **youngest and oldest footballers in FIFA 19** by this code.

{% highlight ruby %}
termuda = fifa19.sort_values('Age', ascending=True)[['Name', 'Age', 'Club', 'Nationality']].head(10)
#untuk mengetahui usia paling tua cukup ganti menjadi ascending=False
{% endhighlight %}

![26ttcl.jpg](https://iili.io/26ttcl.jpg)

![26tDS2.jpg](https://iili.io/26tDS2.jpg)

It can be seen that O. Pérez (45 years old) from Pachuca is the oldest footballer in FIFA. Meanwhile there are several footballers. Meanwhile there are several footballers who are 16 years old which is the youngest age in FIFA 19.

Then, we will see **footballers with longest and shortest tenure for one club**. First, we will take the year contained in the `Joined` column, then we will perform a subtraction operation from the current year (2020) with the year we have obtained from the` Joined` column earlier.

{% highlight ruby %}
import datetime
now = datetime.datetime.now()
fifa19['Join_year'] = fifa19['Joined'].dropna().map(lambda x : x.split(',')[1].split(' ')[1]) #kita akan ambil tahunnya saja
fifa19['Years_of_member'] = (fifa19['Join_year'].dropna().map(lambda x : now.year - int(x))).astype('int')
masa_bakti_panjang = fifa19[['Name', 'Club', 'Years_of_member']].sort_values(by='Years_of_member', ascending=False).head(10)
#untuk mengetahui masa bakti paling sebentar, cukup ganti menjadi ascending=True
{% endhighlight %}

![26DJou.jpg](https://iili.io/26DJou.jpg)

![26D9te.jpg](https://iili.io/26D9te.jpg)

It can be seen that O. Pérez has played for Pachuca for 29 years. Meanwhile there are some players with the shortest tenure for a club, which is only 2 years.

We will see **best left footed and right footed in FIFA 19**

{% highlight ruby %}
fifa19[fifa19['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
#untuk melihat pesepakbola dengan kaki dominan kanan, cukup ganti menjadi Preferred Foot = Left
{% endhighlight %}

![26D2Pj.jpg](https://iili.io/26D2Pj.jpg)

![26DdMb.jpg](https://iili.io/26DdMb.jpg)

It can be seen that Lionel Messi is best left footed and Cristiano Ronaldo is the best right footed in FIFA 19.

We will see **clubs with highest and lowest number of different countries**

{% highlight ruby %}
fifa19.groupby(fifa19['Club'])['Nationality'].nunique().sort_values(ascending=False).head(11)
#untuk melihat pesepakbola dengan jumlah pemain dari negara berbeda paling sedikit, cukup ganti menjadi ascending=True
{% endhighlight %}

![26DIVa.jpg](https://iili.io/26DIVa.jpg)

![26DTiJ.jpg](https://iili.io/26DTiJ.jpg)

It can be seen that footballers who play for Brighton & Hove Albion come from 21 different countries, which the highest in FIFA 19. Meanwhile, there are several clubs that have players from one country only.

Next, we will change `Real Face` and `Preferred Foot` features into numerical data.

{% highlight ruby %}
def face_to_num(data):
    if (data['Real Face'] == 'Yes'):
        return 1
    else:
        return 0
    
def preferred_foot(data):
    if (data['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

fifa19['Real Face'] = fifa19.apply(face_to_num, axis=1)
fifa19['Preferred Foot'] = fifa19.apply(preferred_foot, axis=1)
{% endhighlight %}

### Data Splitting

Now, we will split the data in train data and test data to build Machine Learning and Deep Learning model. But before that, we will take the features we will use for the data training process. We will use datas that have data type `int64` or `float64` only.

{% highlight ruby %}
df = fifa19[['Age', 'Wage', 'Value', 'Special', 'Dribbling', 'Pace', 'Defending', 'Shooting', 'Passing', 'Physical', 'Potential', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Reactions', 'Composure', 'GKDiving', 'GKHandling', 'GKKicking',	'GKPositioning', 'GKReflexes', 'Overall']]
{% endhighlight %}

After that, we will seperate the label/target and features. We use X to define the features and Y to define the label/target. The target that we want to predict from this dataset is `Overall`. The composition for the train data and test data are 80:20. To split the data, we use `train_test_split` from sckit-learn.

{% highlight ruby %}
X = df.drop(['Overall'], axis=1)
y = df['Overall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
{% endhighlight %}

### Machine Learning Model

Next, we will build Machine Learning model using 4 algorithms, **Linear Regression**, **Support Vector Regressor**, **Random Forest Regressor**, and **K-Nearest Neighbors Regressor** algorithms. For the best result, we will do hyperparameter tuning for each algorithm and also we will use `Pipeline` from sckit-learn for the training process.

**Linear Regression**

First, we build the pipeline model which consists of the scaler and the machine learning model. We will use `RobustScaler()` for the scaler.

{% highlight ruby %}
pipeline = Pipeline([('scaler', RobustScaler()), ('model', LinearRegression())])
pipeline.fit(X_train, y_train)
{% endhighlight %}

Then we will use the model to predict using the train data

{% highlight ruby %}
prediksi_linreg_train = pipeline.predict(X_train)
mse_train = mean_squared_error(y_train, prediksi_linreg_train)
r2_train = r2_score(y_train, prediksi_linreg_train)
{% endhighlight %}

Output :

![26DnN1.jpg](https://iili.io/26DnN1.jpg)

Last, we will do data validation by using the model to predict the `Overall` using the test data.

{% highlight ruby %}
prediksi_linreg = pipeline.predict(X_test)
mse = mean_squared_error(y_test, prediksi_linreg)
mae = mean_absolute_error(y_test, prediksi_linreg)
r2 = r2_score(y_test, prediksi_linreg)
{% endhighlight %}

Output :

![26DoDF.jpg](https://iili.io/26DoDF.jpg)

**Support Vector Regressor**

First, we will tune the hyperparameters using `RandomizedSearchCV()`. The parameters to be tuned are as follows.

{% highlight ruby %}
C = [0.001, 0.01, 1.0, 10.0, 100.0, 1000.0]
gamma = [1, 0.1, 0.01, 0.001]

svm_grid = {'model__C' : [1.0, 10.0, 100.0, 1000.0],
            'model__gamma' : [1, 0.1, 0.01, 0.001]}
{% endhighlight %}

Next, we build the pipeline model which consists of the scaler and the machine learning model. Again, we will use `RobustScaler()` as the scaler.

{% highlight ruby %}
pipeline_svm = Pipeline([('scaler', RobustScaler()), ('model', SVR(kernel='rbf'))])
{% endhighlight %}

Then, we do the hyperparameter tuning.

{% highlight ruby %}
svm_random = RandomizedSearchCV(pipeline_svm, svm_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
svm_random.fit(X_train, y_train)
{% endhighlight %}

Check the best parameters.

{% highlight ruby %}
svm_random.best_params_
{% endhighlight %}

The best parameters are **C = 1000** and **gamma = 0.01**. Then, we will add this best parameters into the pipeline model and we will do the training process.

{% highlight ruby %}
pipeline_svm = Pipeline([('scaler', RobustScaler()), ('model', SVR(kernel='rbf', C=1000, gamma=0.01))])
pipeline_svm.fit(X_train, y_train)
{% endhighlight %}

Then we will use the model to predict using the train data

{% highlight ruby %}
prediksi_svm_train = pipeline_svm.predict(X_train)
r2_train = r2_score(y_train, prediksi_svm_train)
mse_train = mean_squared_error(y_train, prediksi_svm_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
{% endhighlight %}

Output :

![26DRlR.jpg](https://iili.io/26DRlR.jpg)

Last, we will do data validation by using the model to predict the `Overall` using the test data

{% highlight ruby %}
prediksi_svm = pipeline_svm.predict(X_test)
mse = mean_squared_error(y_test, prediksi_svm)
mae = mean_absolute_error(y_test, prediksi_svm)
r2 = r2_score(y_test, prediksi_svm)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
{% endhighlight %}

Output :

![26D5Sp.jpg](https://iili.io/26D5Sp.jpg)

**Random Forest Regressor**

First, we will tune the hyperparameters using `RandomizedSearchCV()`. The parameters to be tuned are as follows.

{% highlight ruby %}
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'model__n_estimators' : n_estimators,
               'model__max_features' : max_features, 
               'model__max_depth' : max_depth,
               'model__min_samples_split' : min_samples_split,
               'model__min_samples_leaf' : min_samples_leaf,
               'model__bootstrap' : bootstrap}
{% endhighlight %}

Next, we build the pipeline model which consists of the scaler and the machine learning model. We will use `RobustScaler()` as the scaler.

{% highlight ruby %}
pipeline_rf = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor())])
{% endhighlight %}

Then, we do the hyperparameter tuning.

{% highlight ruby %}
rf_random = RandomizedSearchCV(pipeline_rf, random_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
rf_random.fit(X_train, y_train)
{% endhighlight %}

Check the best parameters

{% highlight ruby %}
rf_random.best_params_
{% endhighlight %}

The best parameters are **bootstrap = True**, **max_depth=20**, **max_features='auto'**, **min_samples_leaf=1**, **min_samples_split=2**, and **n_estimators=1800**. Then, we will add this best parameters into the pipeline model and we will do the training process.

{% highlight ruby %}
pipeline_rf = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor(bootstrap=True, max_depth=20, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=1800))])
pipeline_rf.fit(X_train, y_train)
{% endhighlight %}

Then we will use the model to predict using the train data

{% highlight ruby %}
prediksi_rf_train = pipeline_rf.predict(X_train)
r2_train = r2_score(y_train, prediksi_rf_train)
mse_train = mean_squared_error(y_train, prediksi_rf_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
{% endhighlight %}

Output :

![26DYHN.jpg](https://iili.io/26DYHN.jpg)

Last, we will do data validation by using the model to predict the `Overall` using the test data

{% highlight ruby %}
prediksi_rf = pipeline_rf.predict(X_test)
mse = mean_squared_error(y_test, prediksi_rf)
mae = mean_absolute_error(y_test, prediksi_rf)
r2 = r2_score(y_test, prediksi_rf)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
{% endhighlight %}

Output :

![26DaRI.jpg](https://iili.io/26DaRI.jpg)

**K-Nearest Neighbors Regressor**

First, we will tune the hyperparameters using `RandomizedSearchCV()`. The parameters to be tuned are as follows.

{% highlight ruby %}
n_neighbors = [int(x) for x in np.linspace(start=1, stop=25, num=13)]

knn_grid = {'model__n_neighbors' : n_neighbors}
{% endhighlight %}

Next, we build the pipeline model which consists of the scaler and the machine learning model. We will use `RobustScaler()` as the scaler.

{% highlight ruby %}
pipeline_knn = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsRegressor())])
{% endhighlight %}

Then, we do the hyperparameter tuning.

{% highlight ruby %}
knn_random = RandomizedSearchCV(pipeline_knn, knn_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
knn_random.fit(X_train, y_train)
{% endhighlight %}

Check the best parameters

{% highlight ruby %}
knn_random.best_params_
{% endhighlight %}

The best parameters are **n_neighbors=15**. Then, we will add this best parameters into the pipeline model and we will do the training process.

{% highlight ruby %}
pipeline_knn = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsRegressor(n_neighbors=15))])
pipeline_knn.fit(X_train, y_train)
{% endhighlight %}

Then we will use the model to predict using the train data

{% highlight ruby %}
prediksi_knn_train = pipeline_knn.predict(X_train)
r2_train = r2_score(y_train, prediksi_knn_train)
mse_train = mean_squared_error(y_train, prediksi_knn_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
{% endhighlight %}

Output :

![26DcNt.jpg](https://iili.io/26DcNt.jpg)

Last, we will do data validation by using the model to predict the `Overall` using the test data

{% highlight ruby %}
prediksi_knn = pipeline_knn.predict(X_test)
mse = mean_squared_error(y_test, prediksi_knn)
mae = mean_absolute_error(y_test, prediksi_knn)
r2 = r2_score(y_test, prediksi_knn)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
{% endhighlight %}

Output :

![26DlDX.jpg](https://iili.io/26DlDX.jpg)

From the 4 machine learning models above, Random Forest is the best model with an r2 score 0.99 with rmse 0.43. Therefore, we will save the Random Forest model using `pickle`.

{% highlight ruby %}
#Save model Machine Learning
import pickle

filename = 'best_model.pkl' #Nama filenya
pickle.dump(pipeline_rf, open(filename, 'wb')) #Membuat file model
{% endhighlight %}

### Deep Learning Model

Next, we will build Deep Learning model using Multilayer Perceptron. The libraries that will be used are as follows.

{% highlight ruby %}
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime
{% endhighlight %}

First, We will separate the feature and label/target and scale it using `RobustScaler()`. Then we save the scaler model.

{% highlight ruby %}
X_dl = df.drop(['Overall'], axis=1)
y_dl = df['Overall']

rbst1 = RobustScaler()
rbst2 = RobustScaler()
rbst1 = rbst1.fit(X_dl)
rbst2 = rbst2.fit(df['Overall'].values.reshape(-1, 1))
X_dl = rbst1.transform(X_dl)
y_dl = rbst2.transform(df['Overall'].values.reshape(-1, 1)).flatten()

scalername = 'scaler_feature.pkl' #Nama filenya
pickle.dump(rbst1, open(scalername, 'wb'))

scalername2 = 'scaler_label.pkl' #Nama filenya
pickle.dump(rbst2, open(scalername2, 'wb'))
{% endhighlight %}

Then, we will split the data to be train data and test data with composition of 80:20.

{% highlight ruby %}
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.2, random_state=10)
{% endhighlight %}

Next, we will build the multilayer perceptron architecture

{% highlight ruby %}
model = Sequential()
model.add(Dense(21, input_dim=21, kernel_initializer='uniform', activation='relu')) #memakai 21 neuron dengan input_dim=21 karena ada 21 feature yang dipakai
model.add(Dense(10, kernel_initializer='uniform', activation='relu')) #deeper layer dengan 10 neuron
model.add(Dense(1, kernel_initializer='uniform')) #problem regression sehingga memakai 1 neuron, activation sigmoid

opt = SGD(learning_rate=0.001, momentum=0.9) #optimizer digunakan SGD

model.summary() #melihat summary dari model

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['CosineSimilarity']) #loss yang akan dipakai adalah mean squared error
{% endhighlight %}

This is the summary of the model we've built.

![2yIzR1.jpg](https://iili.io/2yIzR1.jpg)

Next, we save the model and name it `weights_best_only.h5` where we will save the best model only. We using mean_squared_error as loss model so we will save model with the lowest `val_loss`.

{% highlight ruby %}
filepath="weights_best_only.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Tempat dimana log tensorboard akan di
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))
{% endhighlight %}

We train the model.

{% highlight ruby %}
history = model.fit(X_train_dl, y_train_dl, batch_size=32, validation_data=(X_test_dl, y_test_dl), epochs=50, callbacks=callbacks_list, verbose=0)
{% endhighlight %}

We do data validation by using the model to predict the `Overall` using the test data.

{% highlight ruby %}
predict_dl = model.predict(X_test_dl)
predict_dl = predict_dl.flatten()

mse = mean_squared_error(y_test_dl, predict_dl)
mae = mean_absolute_error(y_test_dl, predict_dl)
r2 = r2_score(y_test_dl, predict_dl)
print("MSE (Mean Squared Error)       :", mse)
print("MAE (Mean Absolute Error)      :", mae)
print("r^2 score                      :", r2)
print('RMSE (Root Mean Squared Error) :', np.sqrt(mse))
{% endhighlight %}

Output :

![2yIRWJ.jpg](https://iili.io/2yIRWJ.jpg)

Then, we check the `epoch_loss` and `epoch_cosine_similiarity` graphs using Tensorboard.

{% highlight ruby %}
#load extension jupiter notebook
%reload_ext tensorboard

#load tenserboard
%tensorboard --logdir logs
{% endhighlight %}

**Epoch-Loss Graph**

![2yBGDv.jpg](https://iili.io/2yBGDv.jpg)

**Epoch-Cosine Similiarity Graph**

![2yB1Ra.jpg](https://iili.io/2yB1Ra.jpg)

Based on the `epoch_loss` graph, it can be seen that the model we created has good performance because the graph is sloping and the difference between `loss` and` val_loss` is very small which means that the prediction results are very similar to the expected results with the sweet spot is in epoch 49 with a value of `val_loss` 0.011292. This is also reinforced by the high r2 score which 0.98 and the low rmse which is 0.1. The `epoch_cosine_similiarity` graph also shows a value close to 1, which is 0.9, which means that the value of the test label and the prediction results are similar.

### Model Deployment

The best **Machine Learning** model is with **Random Forest Regressor** algorithm where **r2 score** is obtained with a value of **0.996** and **RMSE** on the Test Data is **0.437**. The **r2 score** is slightly better than the **Deep Learning** model which has **r2 score of 0.981** but the **Deep Learning** model has a better **RMSE** value on the Test Data which is **0.137**

For the model to be deployed, I will deploy **Deep Learning** model because although the r2 score is smaller than the Random Forest, the difference between the r2 score and the Random Forest is only 0.01 with the RMSE difference reaching 0.3 so that if there is a miss prediction on the model, the value of the prediction will not deviate far from the original value.

First, we will load Deep Learning Model and its scalers.

{% highlight ruby %}
from tensorflow.keras.models import load_model

loaded_model = load_model('/content/weights_best_only.h5')
scaler_feature = pickle.load(open(scalername, 'rb'))
scaler_label = pickle.load(open(scalername2, 'rb'))
{% endhighlight %}

After that, we will see the `X_test` and i will take data from index 3207 and i will check the label from index 3207.

{% highlight ruby %}
X_test.head()

#Cek label pada index 3207
y[3207]
{% endhighlight %}

Then, enter the test data in index 3207 in the `test_data` variable. Then we scale the `test_data` with the scaler for the feature. Next, we make predictions with the model we loaded earlier. We will ʻinverse_transform` the results of this prediction so that the prediction results match the labels that have not been scaled. Then we print the prediction results.

{% highlight ruby %}
#Testing data dengan model terbaik ML yaitu Random Forest
test_data = [[31.0,7000.0,2700000.0,1957,73.0,69.0,65.0,62.0,70.0,72.0,73.0,1.0,3.0,3.0,	
68.0,72.0,10.0,12.0,8.0,7.0,14.0]]

test_data1 = scaler_feature.transform(test_data)
predict_model = loaded_model.predict(test_data1)
inv_pred = scaler_label.inverse_transform(predict_model)

print('Overall dari pemain tersebut adalah {}. Keren!'.format(inv_pred[0]))
{% endhighlight %}

Output :

![2yaNAQ.jpg](https://iili.io/2yaNAQ.jpg)

**Web Service**

Next we will deploy the Deep Learning model to the Web Service using Flask. First we will install `flask_ngrok` with `! Pip install flask-ngrok` so that we can run Flask on localhost. Then we import the libraries needed and then create a Flask object and its homepage.

{% highlight ruby %}
from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import jsonify, request

# Membuat sebuah object Flask dan homepage
app = Flask(__name__) 

@app.route("/home")
def home():
    return """<h1>Running Flask on Google Colab!</h1>
              <h2>This is home page!</h2>"""
{% endhighlight %}

Next, we do regression with Machine Learning page.

{% highlight ruby %}
#Regresi dengan halaman Machine Learning

@app.route('/regression', methods=['POST'])
def regression():
  age = float(request.json['Age'])
  wage = float(request.json['Wage'])
  value = float(request.json['Value'])
  special = float(request.json['Special'])
  dribbl_ing = float(request.json['Dribbling'])
  pa_ce = float(request.json['Pace'])
  defend_ing = float(request.json['Defending'])
  shoot_ing = float(request.json['Shooting'])
  pass_ing = float(request.json['Passing'])
  physic_al = float(request.json['Physical'])
  potential = float(request.json['Potential'])
  inter_repu = float(request.json['International Reputation'])
  weak_foot = float(request.json['Weak Foot'])
  skill_move = float(request.json['Skill Moves'])
  reactions = float(request.json['Reactions'])
  composure = float(request.json['Composure'])
  gkd = float(request.json['GKDiving'])
  gkh = float(request.json['GKHandling'])
  gkk = float(request.json['GKKicking'])
  gkp = float(request.json['GKPositioning'])
  gkr = float(request.json['GKReflexes'])

  #Load model
  loaded_model = load_model('/content/weights_best_only.h5')
  scaler_feature = pickle.load(open('scaler_feature.pkl', 'rb'))
  scaler_label = pickle.load(open('scaler_label.pkl', 'rb'))

  data = [[age, wage, value, special, dribbl_ing, pa_ce, defend_ing, shoot_ing, pass_ing, physic_al, potential, inter_repu, weak_foot, skill_move, 
           reactions, composure, gkd, gkh, gkk, gkp, gkr]]
  data1 = scaler_feature.transform(data)
  predict_model = loaded_model.predict(data1)
  inv_pred = scaler_label.inverse_transform(predict_model)

  return jsonify({
      "Player's Overall": str(inv_pred[0][0])
  })
{% endhighlight %}

Run Flask in localhost with `run_with_ngrok()` and we will test the Web Service Using Insomnia.

{% highlight ruby %}
#Jalankan flask di localhost lewat Insomnia
run_with_ngrok(app)

app.run()
{% endhighlight %}

Here is the output of the model we tested in Insomnia:

![2yIYfR.jpg](https://iili.io/2yIYfR.jpg)

Thank you! You can also check my notebook for this project on [Kaggle][kaggle]

[sofifa]: https://sofifa.com/
[kaggle]: https://www.kaggle.com/benedictbayu97/fifa-19-overall-prediction-with-keras