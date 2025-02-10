import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import joblib

# Step 1: Load your dataset
df = pd.read_csv('EPL_20_21.csv')  # ใส่ชื่อไฟล์ของคุณที่นี่

# Step 2: Preprocessing
# กรองข้อมูล โดยเอาเฉพาะตำแหน่ง 'Forward' และ 'Midfielder'
positions_to_keep = ['FW', 'MF', 'MF,FW', 'FW,MF']
df = df[df['Position'].isin(positions_to_keep)]

# Step 3: Create 'Performance' column
def assign_performance(row):
    total_contributions = row['Goals'] + row['Assists']
    if total_contributions >= 12:  # Good
        return 'Good'
    elif total_contributions >= 5:  # Average
        return 'Average'
    else:  # Poor
        return 'Poor'

# เพิ่มคอลัมน์ Performance
df['Performance'] = df.apply(assign_performance, axis=1)

# ตรวจสอบการกระจายตัวของคลาส
print(df['Performance'].value_counts())

# Step 4: Feature Selection
X = df[['Age', 'Matches', 'Starts', 'Mins', 'Goals', 'Assists']]  # Features
y = df['Performance']  # Target

# Step 5: Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 6: Split the dataset into Train and Test sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 8: Cross-validation (Optional)
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

#  แสดงผล Cross-validation Accuracy
print("Cross-validation Accuracy Scores:", cv_scores)
print("Mean Cross-validation Accuracy:", np.mean(cv_scores))

#  พล็อตกราฟ Cross-validation Scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, marker='o', color='b')
plt.title('Cross-validation Scores for Each Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Step 9: Fit the model on the training set
model.fit(X_train, y_train)

# Step 10: Evaluate the model on the test set
y_pred = model.predict(X_test)

#  แสดงผล Accuracy บน Test Data
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Step 11: Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
classes = label_encoder.inverse_transform([1, 0, 2])  # แปลง Label กลับเป็นชื่อคลาส

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Test Data)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 12: Plot Feature Importances
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='teal')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Step 13: Save the model and LabelEncoder
#  บันทึกโมเดล
joblib.dump(model, "model.pkl")

#  บันทึก LabelEncoder
joblib.dump(label_encoder, "label_encoder.pkl")

print(" โมเดลและ LabelEncoder ถูกบันทึกเรียบร้อยแล้ว")