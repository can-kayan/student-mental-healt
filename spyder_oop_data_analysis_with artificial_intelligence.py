import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.linear_model import LinearRegression  # sklearn içinden LinearRegression sınıfını içe aktarın

dataFrame=pd.read_csv("students_mental_health.csv")
dataFrame
dataFrame.info()
dataFrame.describe()
dataColumnX=["Age","Course","Gender","CGPA","Sleep_Quality","Physical_Activity","Diet_Quality","Social_Support","Relationship_Status","Substance_Use","Counseling_Service_Use","Family_History","Chronic_Illness","Extracurricular_Involvement","Semester_Credit_Load","Residence_Type"]

for column in dataColumnX:
    sns.countplot(x=column,data=dataFrame)
    plt.title=f'{column} dağılımı'
    plt.show()
    dataColumnY=["Stress_Level", "Depression_Score" ,"Anxiety_Score", "Financial_Stress"]

for column in dataColumnY:
    sns.countplot(y=column,data=dataFrame)
    plt.title=f'{column} dağılımı'
    plt.show()
    matchingGender={"Female":1,"Male":0}
dataFrame['Gender'] = dataFrame['Gender'].replace(matchingGender)

matchingCourser={
    "Others":0.0,
    "Engineering":.15,
    "Business":0.30,
    "Computer Science":0.45,
    "Medical":0.60,
    "Law":0.75,
}
dataFrame['Course'] = dataFrame['Course'].replace(matchingCourser)

matchingSleepQuality={
    "Good":1.0,
    "Average":0.5,
    "Poor":0.0
}
dataFrame['Sleep_Quality'] = dataFrame['Sleep_Quality'].replace(matchingSleepQuality)

matchingPhysicalActivity={
    "Low":0.0,
    "Moderate":0.5,
    "High":1.0
}
dataFrame['Physical_Activity'] = dataFrame['Physical_Activity'].replace(matchingPhysicalActivity)

matchingDietQuality={
    "Good":1.0,
    "Average":0.5,
    "Poor":0.0
}
dataFrame['Diet_Quality'] = dataFrame['Diet_Quality'].replace(matchingDietQuality)

matchingSocialSupport={
    "Low":1.0,
    "Moderate":0.5,
    "High":0.0
}
dataFrame['Social_Support'] = dataFrame['Social_Support'].replace(matchingSocialSupport)

matchingRelationshipStatus={
    "Married":0.0,
    "In a Relationship":0.5,
    "Single":1.0
}
dataFrame['Relationship_Status'] = dataFrame['Relationship_Status'].replace(matchingRelationshipStatus)

matchingSubstanceUse={
    "Never":0.0,
    "Occasionally":0.5,
    "Frequently":1.0
}
dataFrame['Substance_Use'] = dataFrame['Substance_Use'].replace(matchingSubstanceUse)

matchingCounselingServiceUse={
    "Never":0.0,
    "Occasionally":0.5,
    "Frequently":1.0
}
dataFrame['Counseling_Service_Use'] = dataFrame['Counseling_Service_Use'].replace(matchingCounselingServiceUse)

matchingFamilyHistory={
    "No":0,
    "Yes":1
}
dataFrame['Family_History'] = dataFrame['Family_History'].replace(matchingFamilyHistory)

matchingChronicIllness={
    "No":0,
    "Yes":1
}
dataFrame['Chronic_Illness'] = dataFrame['Chronic_Illness'].replace(matchingChronicIllness)

matchingExtracurricularInvolvement={
    "Low":0.0,
    "Moderate":0.5,
    "High":1.0
}
dataFrame['Extracurricular_Involvement'] = dataFrame['Extracurricular_Involvement'].replace(matchingExtracurricularInvolvement)

matchingResidenceType={
    "With Family":0.0,
    "Off-Campus":0.5,
    "On-Campus":1.0
}
dataFrame['Residence_Type'] = dataFrame['Residence_Type'].replace(matchingResidenceType)
dataFrame
xData=["Age","Course","Gender","Sleep_Quality","Physical_Activity","Diet_Quality","Social_Support","Relationship_Status","Counseling_Service_Use","Family_History","Chronic_Illness","Extracurricular_Involvement","Semester_Credit_Load","Residence_Type"]
yData=["Stress_Level", "Depression_Score" ,"Anxiety_Score", "Financial_Stress"]
dataFrame=dataFrame.dropna()
dataFrame.isnull().sum()
dataFrame.shape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleme veya oluşturma işlemleri buraya eklenebilir.

# Özellik ve hedef değişkenleri seçme
X = dataFrame[xData]

# Veriyi eğitim ve test setlerine ayırma
#for yCol in yData:
X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[0]], test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Performans metriklerini hesaplama
LinearRegression_r2_zeroth = r2_score(y_test, y_pred)
LinearRegression_mae_zeroth = mean_absolute_error(y_test, y_pred)
LinearRegression_mse_zeroth = mean_squared_error(y_test, y_pred)

# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[0]}")
print(f'R-Kare Skoru: {LinearRegression_r2_zeroth}')
print(f'MAE: {LinearRegression_mae_zeroth}')
print(f'MSE: {LinearRegression_mse_zeroth}')

# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': y_test, 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'İlişki Dağılımı - {yData[0]}', y=1.02)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[1]], test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Performans metriklerini hesaplama
LinearRegression_r2_first = r2_score(y_test, y_pred)
LinearRegression_mae_first = mean_absolute_error(y_test, y_pred)
LinearRegression_mse_first = mean_squared_error(y_test, y_pred)

# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[1]}")
print(f'R-Kare Skoru: {LinearRegression_r2_first}')
print(f'MAE: {LinearRegression_mae_first}')
print(f'MSE: {LinearRegression_mse_first}')

# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': y_test, 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'İlişki Dağılımı - {yData[1]}', y=1.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[2]], test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Performans metriklerini hesaplama
LinearRegression_r2_second = r2_score(y_test, y_pred)
LinearRegression_mae_second = mean_absolute_error(y_test, y_pred)
LinearRegression_mse_second = mean_squared_error(y_test, y_pred)

# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[2]}")
print(f'R-Kare Skoru: {LinearRegression_r2_second}')
print(f'MAE: {LinearRegression_mae_second}')
print(f'MSE: {LinearRegression_mse_second}')

# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': y_test, 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'İlişki Dağılımı - {yData[2]}', y=1.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[3]], test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Performans metriklerini hesaplama
LinearRegression_r2_third = r2_score(y_test, y_pred)
LinearRegression_mae_third = mean_absolute_error(y_test, y_pred)
LinearRegression_mse_third = mean_squared_error(y_test, y_pred)

# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[3]}")
print(f'R-Kare Skoru: {LinearRegression_r2_third}')
print(f'MAE: {LinearRegression_mae_third}')
print(f'MSE: {LinearRegression_mse_third}')

# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': y_test, 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'İlişki Dağılımı - {yData[3]}', y=1.02)
plt.show()
from sklearn.tree import DecisionTreeRegressor
X = dataFrame[xData]
#for yCol in yData :

# yData zeroth index
model = DecisionTreeRegressor()
model.fit(X,dataFrame[yData[0]])
y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
# Grafik oluşturma
plt.figure(figsize=(10, 6))
sns.jointplot(x=dataFrame[yData[0]], y=y_pred, kind='reg')
plt.suptitle(f'DecisionTreeRegressor Analizi - {yData[0]}', y=1.02)
plt.show()  
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[0]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'DecisionTreeRegressor Dağılımı - {yData[0]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
DecisionTreeRegressor_r2_zeroth = r2_score(dataFrame[yData[0]], y_pred)
DecisionTreeRegressor_mae_zeroth = mean_absolute_error(dataFrame[yData[0]], y_pred)
DecisionTreeRegressor_mse_zeroth = mean_squared_error(dataFrame[yData[0]], y_pred)
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[0]}")
print(f'R-Kare Skoru: {DecisionTreeRegressor_r2_zeroth}')
print(f'MAE: {DecisionTreeRegressor_mae_zeroth}')
print(f'MSE: {DecisionTreeRegressor_mse_zeroth}')


# yData first index
model = DecisionTreeRegressor()
model.fit(X,dataFrame[yData[1]])
y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
# Grafik oluşturma
plt.figure(figsize=(10, 6))
sns.jointplot(x=dataFrame[yData[1]], y=y_pred, kind='reg')
plt.suptitle(f'DecisionTreeRegressor Analizi - {yData[1]}', y=1.02)
plt.show()  
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[1]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'DecisionTreeRegressor Dağılımı - {yData[1]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
DecisionTreeRegressor_r2_first = r2_score(dataFrame[yData[1]], y_pred)
DecisionTreeRegressor_mae_first = mean_absolute_error(dataFrame[yData[1]], y_pred)
DecisionTreeRegressor_mse_first = mean_squared_error(dataFrame[yData[1]], y_pred)
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[1]}")
print(f'R-Kare Skoru: {DecisionTreeRegressor_r2_first}')
print(f'MAE: {DecisionTreeRegressor_mae_first}')
print(f'MSE: {DecisionTreeRegressor_mse_first}')


# yData second index
model = DecisionTreeRegressor()
model.fit(X,dataFrame[yData[2]])
y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
# Grafik oluşturma
plt.figure(figsize=(10, 6))
sns.jointplot(x=dataFrame[yData[2]], y=y_pred, kind='reg')
plt.suptitle(f'DecisionTreeRegressor Analizi - {yData[2]}', y=1.02)
plt.show()  
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[2]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'DecisionTreeRegressor Dağılımı - {yData[2]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
DecisionTreeRegressor_r2_second = r2_score(dataFrame[yData[2]], y_pred)
DecisionTreeRegressor_mae_second = mean_absolute_error(dataFrame[yData[2]], y_pred)
DecisionTreeRegressor_mse_second = mean_squared_error(dataFrame[yData[2]], y_pred)
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[2]}")
print(f'R-Kare Skoru: {DecisionTreeRegressor_r2_second}')
print(f'MAE: {DecisionTreeRegressor_mae_second}')
print(f'MSE: {DecisionTreeRegressor_mse_second}')


# yData thrid index
model = DecisionTreeRegressor()
model.fit(X,dataFrame[yData[3]])
y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
# Grafik oluşturma
plt.figure(figsize=(10, 6))
sns.jointplot(x=dataFrame[yData[3]], y=y_pred, kind='reg')
plt.suptitle(f'DecisionTreeRegressor Analizi - {yData[3]}', y=1.02)
plt.show()  
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[3]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'DecisionTreeRegressor Dağılımı - {yData[3]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
DecisionTreeRegressor_r2_third = r2_score(dataFrame[yData[3]], y_pred)
DecisionTreeRegressor_mae_third = mean_absolute_error(dataFrame[yData[3]], y_pred)
DecisionTreeRegressor_mse_third = mean_squared_error(dataFrame[yData[3]], y_pred)
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[3]}")
print(f'R-Kare Skoru: {DecisionTreeRegressor_r2_third}')
print(f'MAE: {DecisionTreeRegressor_mae_third}')
print(f'MSE: {DecisionTreeRegressor_mse_third}')
from sklearn.ensemble import RandomForestRegressor

X = dataFrame[xData]
#for yCol in yData :


model = RandomForestRegressor(n_estimators=100)
model.fit(X, dataFrame[yData[0]])
y_pred = model.predict(X)
# Grafik oluşturma
residuals = dataFrame[yData[0]] - y_pred
plt.figure(figsize=(10, 6))
sns.barplot(x=dataFrame[yData[0]], y=residuals, color='blue')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.suptitle(f'RandomForestRegressor Analizi - {yData[0]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
RandomForestRegressor_r2_zeroth = r2_score(dataFrame[yData[0]], y_pred)
RandomForestRegressor_mae_zeroth = mean_absolute_error(dataFrame[yData[0]], y_pred)
RandomForestRegressor_mse_zeroth = mean_squared_error(dataFrame[yData[0]], y_pred)
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[0]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'RandomForestRegressor Dağılımı - {yData[0]}', y=1.02)
plt.show()
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[0]}")
print(f'R-Kare Skoru: {RandomForestRegressor_r2_zeroth}')
print(f'MAE: {RandomForestRegressor_mae_zeroth}')
print(f'MSE: {RandomForestRegressor_mse_zeroth}')


model = RandomForestRegressor(n_estimators=100)
model.fit(X, dataFrame[yData[1]])
y_pred = model.predict(X)
# Grafik oluşturma
residuals = dataFrame[yData[1]] - y_pred
plt.figure(figsize=(10, 6))
sns.barplot(x=dataFrame[yData[0]], y=residuals, color='blue')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.suptitle(f'RandomForestRegressor Analizi - {yData[1]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
RandomForestRegressor_r2_first = r2_score(dataFrame[yData[1]], y_pred)
RandomForestRegressor_mae_first = mean_absolute_error(dataFrame[yData[1]], y_pred)
RandomForestRegressor_mse_first = mean_squared_error(dataFrame[yData[1]], y_pred)
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[0]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'RandomForestRegressor Dağılımı - {yData[1]}', y=1.02)
plt.show()
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[1]}")
print(f'R-Kare Skoru: {RandomForestRegressor_r2_first}')
print(f'MAE: {RandomForestRegressor_mae_first}')
print(f'MSE: {RandomForestRegressor_mse_first}')



model = RandomForestRegressor(n_estimators=100)
model.fit(X, dataFrame[yData[2]])
y_pred = model.predict(X)
# Grafik oluşturma
residuals = dataFrame[yData[2]] - y_pred
plt.figure(figsize=(10, 6))
sns.barplot(x=dataFrame[yData[2]], y=residuals, color='blue')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.suptitle(f'RandomForestRegressor Analizi - {yData[2]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
RandomForestRegressor_r2_second = r2_score(dataFrame[yData[2]], y_pred)
RandomForestRegressor_mae_second = mean_absolute_error(dataFrame[yData[2]], y_pred)
RandomForestRegressor_mse_second = mean_squared_error(dataFrame[yData[2]], y_pred)
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[2]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'RandomForestRegressor Dağılımı - {yData[2]}', y=1.02)
plt.show()
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[2]}")
print(f'R-Kare Skoru: {RandomForestRegressor_r2_second}')
print(f'MAE: {RandomForestRegressor_mae_second}')
print(f'MSE: {RandomForestRegressor_mse_second}')



model = RandomForestRegressor(n_estimators=100)
model.fit(X, dataFrame[yData[3]])
y_pred = model.predict(X)
# Grafik oluşturma
residuals = dataFrame[yData[3]] - y_pred
plt.figure(figsize=(10, 6))
sns.barplot(x=dataFrame[yData[3]], y=residuals, color='blue')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.suptitle(f'RandomForestRegressor Analizi - {yData[3]}', y=1.02)
plt.show()
# Performans metriklerini hesaplama
RandomForestRegressor_r2_third = r2_score(dataFrame[yData[3]], y_pred)
RandomForestRegressor_mae_third = mean_absolute_error(dataFrame[yData[3]], y_pred)
RandomForestRegressor_mse_third = mean_squared_error(dataFrame[yData[3]], y_pred)
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[3]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'RandomForestRegressor Dağılımı - {yData[3]}', y=1.02)
plt.show()
# Sonuçları ekrana yazdırma
print(f"\nHedef Değişken: {yData[3]}")
print(f'R-Kare Skoru: {RandomForestRegressor_r2_third}')
print(f'MAE: {RandomForestRegressor_mae_third}')
print(f'MSE: {RandomForestRegressor_mse_third}')
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

model = KNeighborsRegressor(n_neighbors=3)
X = dataFrame[xData]
#for yCol in yData :
# Modeli eğitme
model.fit(X, dataFrame[yData[0]])
# Tahmin yapma
y_pred = model.predict(X)
# Performans metriklerini hesaplama
KNeighborsRegressor_r2_zeroth = r2_score(dataFrame[yData[0]], y_pred)
KNeighborsRegressor_mae_zeroth = mean_absolute_error(dataFrame[yData[0]], y_pred)
KNeighborsRegressor_mse_zeroth = mean_squared_error(dataFrame[yData[0]], y_pred)
# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[0]}")
print(f'R-Kare Skoru: {KNeighborsRegressor_r2_zeroth}')
print(f'MAE: {KNeighborsRegressor_mae_zeroth}')
print(f'MSE: {KNeighborsRegressor_mse_zeroth}')
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[0]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'KNeighborsRegressor Dağılımı - {yData[0]}', y=1.02)
plt.show()



model = KNeighborsRegressor(n_neighbors=3)
X = dataFrame[xData]
#for yCol in yData :
# Modeli eğitme
model.fit(X, dataFrame[yData[1]])
# Tahmin yapma
y_pred = model.predict(X)
# Performans metriklerini hesaplama
KNeighborsRegressor_r2_first = r2_score(dataFrame[yData[1]], y_pred)
KNeighborsRegressor_mae_first = mean_absolute_error(dataFrame[yData[1]], y_pred)
KNeighborsRegressor_mse_first = mean_squared_error(dataFrame[yData[1]], y_pred)
# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[1]}")
print(f'R-Kare Skoru: {KNeighborsRegressor_r2_first}')
print(f'MAE: {KNeighborsRegressor_mae_first}')
print(f'MSE: {KNeighborsRegressor_mse_first}')
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[1]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'KNeighborsRegressor Dağılımı - {yData[1]}', y=1.02)
plt.show()




model = KNeighborsRegressor(n_neighbors=3)
X = dataFrame[xData]
#for yCol in yData :
# Modeli eğitme
model.fit(X, dataFrame[yData[2]])
# Tahmin yapma
y_pred = model.predict(X)
# Performans metriklerini hesaplama
KNeighborsRegressor_r2_second= r2_score(dataFrame[yData[2]], y_pred)
KNeighborsRegressor_mae_second = mean_absolute_error(dataFrame[yData[2]], y_pred)
KNeighborsRegressor_mse_second = mean_squared_error(dataFrame[yData[2]], y_pred)
# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[2]}")
print(f'R-Kare Skoru: {KNeighborsRegressor_r2_second}')
print(f'MAE: {KNeighborsRegressor_mae_second}')
print(f'MSE: {KNeighborsRegressor_mse_second}')
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[2]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'KNeighborsRegressor Dağılımı - {yData[2]}', y=1.02)
plt.show()





model = KNeighborsRegressor(n_neighbors=3)
X = dataFrame[xData]
#for yCol in yData :
# Modeli eğitme
model.fit(X, dataFrame[yData[3]])
# Tahmin yapma
y_pred = model.predict(X)
# Performans metriklerini hesaplama
KNeighborsRegressor_r2_third = r2_score(dataFrame[yData[3]], y_pred)
KNeighborsRegressor_mae_third = mean_absolute_error(dataFrame[yData[3]], y_pred)
KNeighborsRegressor_mse_third = mean_squared_error(dataFrame[yData[3]], y_pred)
# Sonuçları ayrı ayrı yazdırma
print(f"\nHedef Değişken: {yData[3]}")
print(f'R-Kare Skoru: {KNeighborsRegressor_r2_third}')
print(f'MAE: {KNeighborsRegressor_mae_third}')
print(f'MSE: {KNeighborsRegressor_mse_third}')
# Grafik oluştur
results_df = pd.DataFrame({'Gerçek Değerler': dataFrame[yData[3]], 'Tahmin Edilen Değerler': y_pred})
sns.pairplot(results_df, height=2, aspect=1.5, kind="reg", diag_kind="kde")
plt.suptitle(f'KNeighborsRegressor Dağılımı - {yData[3]}', y=1.02)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#for yCol in yData :
X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[0]], test_size=0.2, random_state=42)
# KNN modelini oluşturun ve eğitin
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = knn_model.predict(X_test)
# Model doğruluğunu değerlendirin
KNeighborsClassifier_accuracy_zeroth = accuracy_score(y_test, y_pred)
print(f"KNN Model Doğruluğu: {KNeighborsClassifier_accuracy_zeroth}")
# Sınıflandırma raporu
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'KNeighborsClassifier Hata Matrisi - {yData[0]}', y=1.02)
plt.show()




X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[1]], test_size=0.2, random_state=42)
# KNN modelini oluşturun ve eğitin
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = knn_model.predict(X_test)
# Model doğruluğunu değerlendirin
KNeighborsClassifier_accuracy_first = accuracy_score(y_test, y_pred)
print(f"KNN Model Doğruluğu: {KNeighborsClassifier_accuracy_first}")
# Sınıflandırma raporu
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'KNeighborsClassifier Hata Matrisi - {yData[1]}', y=1.02)
plt.show()





X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[2]], test_size=0.2, random_state=42)
# KNN modelini oluşturun ve eğitin
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = knn_model.predict(X_test)
# Model doğruluğunu değerlendirin
KNeighborsClassifier_accuracy_second = accuracy_score(y_test, y_pred)
print(f"KNN Model Doğruluğu: {KNeighborsClassifier_accuracy_second}")
# Sınıflandırma raporu
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'KNeighborsClassifier Hata Matrisi - {yData[2]}', y=1.02)
plt.show()





X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[3]], test_size=0.2, random_state=42)
# KNN modelini oluşturun ve eğitin
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = knn_model.predict(X_test)
# Model doğruluğunu değerlendirin
KNeighborsClassifier_accuracy_third = accuracy_score(y_test, y_pred)
print(f"KNN Model Doğruluğu: {KNeighborsClassifier_accuracy_third}")
# Sınıflandırma raporu
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'KNeighborsClassifier Hata Matrisi - {yData[3]}', y=1.02)
plt.show()


from sklearn.naive_bayes import GaussianNB

#for yCol in yData :
X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[0]], test_size=0.25, random_state=42)
# Naive Bayes modeli oluşturma ve eğitme
model = GaussianNB()
model.fit(X_train, y_train)
# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)
# Doğruluk (accuracy) değerini hesaplama
GaussianNB_accuracy_zeroth = accuracy_score(y_test, y_pred)
print(f"Doğruluk ({yData[0]}): {GaussianNB_accuracy_zeroth}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[0]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'GaussianNB Hata Matrisi - {yData[0]}', y=1.02)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[1]], test_size=0.25, random_state=42)
# Naive Bayes modeli oluşturma ve eğitme
model = GaussianNB()
model.fit(X_train, y_train)
# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)
# Doğruluk (accuracy) değerini hesaplama
GaussianNB_accuracy_first = accuracy_score(y_test, y_pred)
print(f"Doğruluk ({yData[1]}): {GaussianNB_accuracy_first}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[1]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'GaussianNB Hata Matrisi - {yData[1]}', y=1.02)
plt.show()





X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[2]], test_size=0.25, random_state=42)
# Naive Bayes modeli oluşturma ve eğitme
model = GaussianNB()
model.fit(X_train, y_train)
# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)
# Doğruluk (accuracy) değerini hesaplama
GaussianNB_accuracy_second = accuracy_score(y_test, y_pred)
print(f"Doğruluk ({yData[2]}): {GaussianNB_accuracy_second}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[2]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'GaussianNB Hata Matrisi - {yData[2]}', y=1.02)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[3]], test_size=0.25, random_state=42)
# Naive Bayes modeli oluşturma ve eğitme
model = GaussianNB()
model.fit(X_train, y_train)
# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)
# Doğruluk (accuracy) değerini hesaplama
GaussianNB_accuracy_third = accuracy_score(y_test, y_pred)
print(f"Doğruluk ({yData[3]}): {GaussianNB_accuracy_third}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[3]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'GaussianNB Hata Matrisi - {yData[3]}', y=1.02)
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#for yCol in yData:
X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[0]], test_size=0.2, random_state=42)
# RandomForestClassifier modelini oluşturun ve eğitin
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = rf_model.predict(X_test)
# Model doğruluğunu değerlendirin
RandomForestClassifier_accuracy_zeroth = accuracy_score(y_test, y_pred)
print(f"RandomForestClassifier Model Doğruluğu ({yData[0]}): {RandomForestClassifier_accuracy_zeroth}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[0]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'RandomForestClassifier Hata Matrisi - {yData[0]}', y=1.02)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[1]], test_size=0.2, random_state=42)
# RandomForestClassifier modelini oluşturun ve eğitin
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = rf_model.predict(X_test)
# Model doğruluğunu değerlendirin
RandomForestClassifier_accuracy_first = accuracy_score(y_test, y_pred)
print(f"RandomForestClassifier Model Doğruluğu ({yData[1]}): {RandomForestClassifier_accuracy_first}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[1]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'RandomForestClassifier Hata Matrisi - {yData[1]}', y=1.02)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[2]], test_size=0.2, random_state=42)
# RandomForestClassifier modelini oluşturun ve eğitin
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = rf_model.predict(X_test)
# Model doğruluğunu değerlendirin
RandomForestClassifier_accuracy_second = accuracy_score(y_test, y_pred)
print(f"RandomForestClassifier Model Doğruluğu ({yData[2]}): {RandomForestClassifier_accuracy_second}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[2]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'RandomForestClassifier Hata Matrisi - {yData[2]}', y=1.02)
plt.show()




X_train, X_test, y_train, y_test = train_test_split(X, dataFrame[yData[3]], test_size=0.2, random_state=42)
# RandomForestClassifier modelini oluşturun ve eğitin
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Tahminleri yapın
y_pred = rf_model.predict(X_test)
# Model doğruluğunu değerlendirin
RandomForestClassifier_accuracy_third = accuracy_score(y_test, y_pred)
print(f"RandomForestClassifier Model Doğruluğu ({yData[3]}): {RandomForestClassifier_accuracy_third}")
# Sınıflandırma raporu
print(f"Sınıflandırma Raporu ({yData[3]}):\n", classification_report(y_test, y_pred))
# Hata matrisini görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.suptitle(f'RandomForestClassifier Hata Matrisi - {yData[3]}', y=1.02)
plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optimal küme sayısını belirleme (elbow method)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Elbow method grafiği
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.suptitle('Elbow Method - Optimal Küme Sayısı')
plt.xlabel('Küme Sayısı')
plt.ylabel('İnertia (Toplam Kare Hata)')
plt.show()

# Optimal küme sayısına göre model oluşturma ve eğitme
optimal_k = 3  # Elbow method grafiğine bakarak uygun küme sayısını belirle
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Küme etiketlerini al
KMeans_cluster_labels = kmeans.labels_

# Sonuçları yazdır
print("Küme Etiketleri:", KMeans_cluster_labels)

# Küme merkezlerini inceleme
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Küme Merkezleri:", cluster_centers)

# Veriyi ve küme merkezlerini görselleştirme
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=KMeans_cluster_labels, palette='viridis', alpha=0.8)
sns.scatterplot(x=cluster_centers[:, 0], y=cluster_centers[:, 1], marker='X', s=200, color='red', label='Küme Merkezleri')
plt.suptitle('K-Means Kümeleme Sonuçları')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.show()




#Model isimleri ve r2 skorları 
model_names_r2 = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'K-Nearest Neighbors Regressor']

r2_scores = [LinearRegression_r2_zeroth, DecisionTreeRegressor_r2_zeroth, RandomForestRegressor_r2_zeroth, KNeighborsRegressor_r2_zeroth]
plt.figure(figsize=(10, 6))
sns.barplot(x=r2_scores, y=model_names_r2, palette='viridis')
plt.suptitle(f'R-Kare Skorları Karşılaştırması - {yData[0]}')
plt.ylabel('R-Kare Skoru')
plt.show()

r2_scores = [LinearRegression_r2_first, DecisionTreeRegressor_r2_first, RandomForestRegressor_r2_first, KNeighborsRegressor_r2_first]
plt.figure(figsize=(10, 6))
sns.barplot(x=r2_scores, y=model_names_r2, palette='viridis')
plt.suptitle(f'R-Kare Skorları Karşılaştırması - {yData[1]}')
plt.ylabel('R-Kare Skoru')
plt.show()

r2_scores = [LinearRegression_r2_second, DecisionTreeRegressor_r2_second, RandomForestRegressor_r2_second, KNeighborsRegressor_r2_second]
plt.figure(figsize=(10, 6))
sns.barplot(x=r2_scores, y=model_names_r2, palette='viridis')
plt.suptitle(f'R-Kare Skorları Karşılaştırması - {yData[2]}')
plt.ylabel('R-Kare Skoru')
plt.show()

r2_scores = [LinearRegression_r2_third, DecisionTreeRegressor_r2_third, RandomForestRegressor_r2_third, KNeighborsRegressor_r2_third]
plt.figure(figsize=(10, 6))
sns.barplot(x=r2_scores, y=model_names_r2, palette='viridis')
plt.suptitle(f'R-Kare Skorları Karşılaştırması - {yData[3]}')
plt.ylabel('R-Kare Skoru')
plt.show()




#Model isimleri ve mae skorları 
model_names_mae = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'K-Nearest Neighbors Regressor']


mae_scores = [LinearRegression_mae_zeroth, DecisionTreeRegressor_mae_zeroth, RandomForestRegressor_mae_zeroth, KNeighborsRegressor_mae_zeroth]
plt.figure(figsize=(10, 6))
sns.barplot(x=mae_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mae Karşılaştırması - {yData[0]}')
plt.ylabel('mae Skoru')
plt.show()





mae_scores = [LinearRegression_mae_first, DecisionTreeRegressor_mae_first, RandomForestRegressor_mae_first, KNeighborsRegressor_mae_first]
plt.figure(figsize=(10, 6))
sns.barplot(x=mae_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mae Karşılaştırması - {yData[1]}')
plt.ylabel('mae Skoru')
plt.show()





mae_scores = [LinearRegression_mae_second, DecisionTreeRegressor_mae_second, RandomForestRegressor_mae_second, KNeighborsRegressor_mae_second]
plt.figure(figsize=(10, 6))
sns.barplot(x=mae_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mae Karşılaştırması - {yData[2]}')
plt.ylabel('mae Skoru')
plt.show()





mae_scores = [LinearRegression_mae_third, DecisionTreeRegressor_mae_third, RandomForestRegressor_mae_third, KNeighborsRegressor_mae_third]
plt.figure(figsize=(10, 6))
sns.barplot(x=mae_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mae Karşılaştırması - {yData[3]}')
plt.ylabel('mae Skoru')
plt.show()


#Model isimleri ve mse skorları 
model_names_mae = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'K-Nearest Neighbors Regressor']


mse_scores = [LinearRegression_mse_zeroth, DecisionTreeRegressor_mse_zeroth, RandomForestRegressor_mse_zeroth, KNeighborsRegressor_mse_zeroth]
plt.figure(figsize=(10, 6))
sns.barplot(x=mse_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mae Karşılaştırması - {yData[0]}')
plt.ylabel('mse Skoru')
plt.show()

mse_scores = [LinearRegression_mse_first, DecisionTreeRegressor_mse_first, RandomForestRegressor_mse_first, KNeighborsRegressor_mse_first]
plt.figure(figsize=(10, 6))
sns.barplot(x=mse_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mse Karşılaştırması - {yData[1]}')
plt.ylabel('mse Skoru')
plt.show()


mse_scores = [LinearRegression_mse_second, DecisionTreeRegressor_mse_second, RandomForestRegressor_mse_second, KNeighborsRegressor_mse_second]
plt.figure(figsize=(10, 6))
sns.barplot(x=mse_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mse Karşılaştırması - {yData[2]}')
plt.ylabel('mse Skoru')
plt.show()





mse_scores = [LinearRegression_mse_third, DecisionTreeRegressor_mse_third, RandomForestRegressor_mse_third, KNeighborsRegressor_mse_third]
plt.figure(figsize=(10, 6))
sns.barplot(x=mse_scores, y=model_names_mae, palette='viridis')
plt.suptitle(f'mse Karşılaştırması - {yData[3]}')
plt.ylabel('mse Skoru')
plt.show()





LinearRegression_r2_scores = [LinearRegression_r2_zeroth, LinearRegression_r2_first, LinearRegression_r2_second, LinearRegression_r2_third]
plt.figure(figsize=(10, 6))
sns.barplot(x=yData , y=LinearRegression_r2_scores, palette='viridis')
plt.suptitle(f'Linear Regression Karşılaştırması')
plt.ylabel('r2 Skoru')
plt.show()

DecisionTreeRegressor_r2_scores = [DecisionTreeRegressor_r2_zeroth, DecisionTreeRegressor_r2_first, DecisionTreeRegressor_r2_second, DecisionTreeRegressor_r2_third]
plt.figure(figsize=(10, 6))
sns.barplot(x=yData , y=DecisionTreeRegressor_r2_scores, palette='viridis')
plt.suptitle(f'Decision Tree Regressor Karşılaştırması')
plt.ylabel('r2 Skoru')
plt.show()

RandomForestRegressor_r2_scores = [RandomForestRegressor_r2_zeroth, RandomForestRegressor_r2_first, RandomForestRegressor_r2_second, RandomForestRegressor_r2_third]
plt.figure(figsize=(10, 6))
sns.barplot(x=yData , y=RandomForestRegressor_r2_scores, palette='viridis')
plt.suptitle(f'Random Forest Regressor Karşılaştırması')
plt.ylabel('r2 Skoru')
plt.show()

KNeighborsRegressor_r2_scores = [KNeighborsRegressor_r2_zeroth, KNeighborsRegressor_r2_first, KNeighborsRegressor_r2_second, KNeighborsRegressor_r2_third]
plt.figure(figsize=(10, 6))
sns.barplot(x=yData , y=KNeighborsRegressor_r2_scores, palette='viridis')
plt.suptitle(f'K Neighbors Regressor Karşılaştırması')
plt.ylabel('r2 Skoru')
plt.show()






model_names_classifiers = ['KNeighbors Classifier', 'GaussianNB', 'Random Forest Classifier']

 
accuracy_scores = [KNeighborsClassifier_accuracy_zeroth, GaussianNB_accuracy_zeroth, RandomForestClassifier_accuracy_zeroth]
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracy_scores, y=model_names_classifiers, palette='viridis')
plt.suptitle(f'classifiers Karşılaştırması - {yData}')
plt.ylabel('classifiers Skoru')
plt.show()




























