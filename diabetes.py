import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
from  sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
########Görev 1 : Keşifçi Veri Analizi

#Adım 1: Genel resmi inceleyiniz.

df = pd.read_csv("DERSLER/Feature Engineering/case_study_diabetes/diabetes.csv")
df.head()

df.isnull().sum()
#null değer yok.
df.shape
df.describe().T

df.dtypes #full num gibi gözüküyor.

df.info()

#Adım 2: Numerik ve kategorik değişkenleri yakalayınız

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """

       Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
       Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

       Parameters
       ------
           dataframe: dataframe
                   Değişken isimleri alınmak istenilen dataframe
           cat_th: int, optional
                   numerik fakat kategorik olan değişkenler için sınıf eşik değeri
           car_th: int, optinal
                   kategorik fakat kardinal değişkenler için sınıf eşik değeri

       Returns
       ------
           cat_cols: list
                   Kategorik değişken listesi
           num_cols: list
                   Numerik değişken listesi
           cat_but_car: list
                   Kategorik görünümlü kardinal değişken listesi

       Examples
       ------
           import seaborn as sns
           df = sns.load_dataset("iris")
           print(grab_col_names(df))


       Notes
       ------
           cat_cols + num_cols + cat_but_car = toplam değişken sayısı
           num_but_cat cat_cols'un içerisinde.
           Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

       """
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" ]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]


    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
#9 column'un 1i cat 8i num. cat_cols="Outcome"
cat_cols

# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


        for col in age_cols:
            num_summary(df,col,True)

num_summary(df,num_cols,plot=True)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df,"Outcome")


# Adım 4:Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, colname, q1 =0.25, q3= 0.75):
    quartile1 = dataframe[colname].quantile(q1)
    quartile3 = dataframe[colname].quantile(q3)
    interquartile = quartile3 - quartile1
    up_limit = quartile3 + interquartile * 1.5
    low_limit = quartile1 - interquartile * 1.5
    return  low_limit, up_limit
def check_outlier(dataframe,colname):
    low, up = outlier_thresholds(dataframe,colname)
    if dataframe[(dataframe[colname] < low ) | (dataframe[colname] > up)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe,colname):
    low, up = outlier_thresholds(dataframe,colname)
    df.loc[(df[colname] > up), colname] = up
    df.loc[(df[colname] < low), colname] = low

for col in num_cols:
    print(col, check_outlier(df,col))
    #bütün numlarda aykırı gözlem var.

for col in num_cols:
    replace_with_thresholds(df,col)

#aykırı gözlemleri replace ettik.


#Adım 6: Eksik gözlem analizi yapınız.

df.isnull().sum()
df.describe().T

def missing_values_table(dataframe,na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df,True)

#eksik gözlem yok


#Adım 7: Korelasyon analizi yapınız.

corr_matrix = df.corr()
corr_matrix


########## Görev 2 : Feature Engineering
# Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumudikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
df.describe().T
#SkinThickness, Insulin bu değerlerin min degeri 0. bu değerler 0 olamaz. aslında eksik deger fakat 0 girilmiş olabilir.
#ama pregnancy ve outcome 0 olabilir. onlar normal.

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
#bu degeler 0 ama aslında nan. bunları nan a cevirelim.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

missing_values_table(df)

def missing_vs_target(dataframe,target,na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(),1,0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df,"Outcome",zero_columns)

#outliersa yukarıda bakıp onları baskılamıştık
#missing valuelar vs nonmiss valuelarda da target meanle
#aralarındaki korelasyon acısından bir fark göremedik.


#bu null degerleri dolduralım....

for col in zero_columns:
    df.loc[df[col].isnull(),col] = df[col].median()

# Adım 2: Yeni değişkenler oluşturunuz.
df.head()
#yaşa göre bi sınıf cıkaralım ortaya.
df.head()
df["Age"].describe().T
df["NEW_AGE_CAT"].nunique()

df.loc[(df["Age"] < 45), "NEW_AGE_CAT"] ="mature"
df.loc[(df["Age"] > 45), "NEW_AGE_CAT"] ="senior"

#bmi chartına göre 18.5 is underweight, 18.5 to 24.9 is normal, 24.9 to 29.9 is Overweight, and over 30 is obese
df.head()
df.loc[(df["BMI"]> 0) & (df["BMI"] < 18.5), "NEW_BMI_CAT"] ="underweight"
df.loc[(df["BMI"]>= 18.5) & (df["BMI"] < 24.9), "NEW_BMI_CAT"] ="normal"
df.loc[(df["BMI"]>= 24.9) & (df["BMI"] <= 29.99), "NEW_BMI_CAT"] ="overweight"
df.loc[(df["BMI"]>= 30), "NEW_BMI_CAT"] ="obese"

df.head()
df["NEW_BMI_CAT"].nunique()

#glikoz değeri için de bir ayırım var.
df["Glucose"].describe().T
df.loc[(df["Glucose"]> 0) & (df["Glucose"] < 140), "NEW_GLUCOSE_CAT"] ="normal"
df.loc[(df["Glucose"]>= 140) & (df["Glucose"] < 200), "NEW_GLUCOSE_CAT"] ="prediabetic"
df.loc[(df["Glucose"]>= 200), "NEW_GLUCOSE_CAT"] ="diabetic"

df["NEW_GLUCOSE_CAT"].nunique()

#diabetik olan yokmuş. maks deger 199du cunku

#bu yaptıklarımızı yaşa göre kıralım.

df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_CAT"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_CAT"] = "obesesenior"


df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_CAT"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_CAT"] = "highsenior"


df.head()

# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
df.shape #14 columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#####   label encoding yapalım...

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

#outcome bagımlı degiskenim ama 0ve1 zaten cıkarmaya gerek yok binary colstan.

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df,col)

df.head()

##### one hot encoding...

def one_hot_encoder(dataframe,categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10>= df[col].nunique() > 2]


df = one_hot_encoder(df, ohe_cols, drop_first=True)

df.head()

##### Rare Encoding var mı ona bakalım....
cat_cols,num_cols, cat_but_car = grab_col_names(df)


def rare_analyser(dataframe,target,cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df,"Outcome",cat_cols)

#bazı değerler gerçekten gereksiz. ama silmemeyi tercih ediyoruz.



# Adım 4: Numerik değişkenler için standartlaştırma yapınız.Adım
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


# 5: Model oluşturunuz.

df.shape #24 tane columnla yola devam ediyoruz..

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=33)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#0.7619 başarı oranımız....

