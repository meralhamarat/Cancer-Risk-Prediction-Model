import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Veriyi yükleme
data = pd.read_csv("cancer_risk_data.csv")

# Cinsiyet sütununu sayısal yapalım
data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})

# Özellik ve hedef ayrımı
X = data.drop('CancerRisk', axis=1)
y = data['CancerRisk']

# Ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model eğitimi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test verisinde tahmin yap
y_pred = model.predict(X_test)

# Doğruluk skoru
acc = accuracy_score(y_test, y_pred)
print("Doğruluk Oranı:", acc)

# Feature importance (Özellik önemi)
importances = model.feature_importances_
features = X.columns

# Özellik önemini çizdirme
plt.figure(figsize=(8,5))
plt.bar(features, importances, color='lightcoral')
plt.title('Özelliklerin Modeldeki Önemi')
plt.ylabel('Önem Skoru')
plt.ylim(0, max(importances) + 0.05)
plt.xticks(rotation=45)
plt.show()

# Confusion matrix çizdirme
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
