import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('student_performance_2024.csv')

# Encode categorical grades
grade_encoder = LabelEncoder()
df['Last_Year_Grade_Encoded'] = grade_encoder.fit_transform(df['Last_Year_Grade'])

# Encode final result (Pass/Fail)
result_encoder = LabelEncoder()
df['Final_Result_Encoded'] = result_encoder.fit_transform(df['Final_Result'])

# Define input features and target
features = ['Internal_1', 'Internal_2', 'Attendance', 'Last_Year_Grade_Encoded']
target = 'Final_Result_Encoded'

# Train/test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict all students' results
df['Predicted_Result'] = model.predict(X)
df['Predicted_Result_Label'] = df['Predicted_Result'].apply(lambda x: 'Pass' if x == 1 else 'Fail')

# Generate actionable recommendations
def generate_recommendation(row):
    recs = []
    if row['Attendance'] < 75:
        recs.append("📉 Improve attendance")
    if (row['Internal_1'] + row['Internal_2']) / 2 < 12:
        recs.append("📚 Improve internal scores")
    if row['Last_Year_Grade'] in ['C', 'D']:
        recs.append("🔁 Needs academic support")
    return ', '.join(recs) if recs else "✅ Good standing"

df['Recommendation'] = df.apply(generate_recommendation, axis=1)

# Save the results to a new CSV
df.to_csv('predicted_student_performance.csv', index=False)
print("✅ Model completed. Output saved as 'predicted_student_performance.csv'")
