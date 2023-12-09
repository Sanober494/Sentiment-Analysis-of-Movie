# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# Read the data provided
df = pd.read_csv("/kaggle/input/newdataset/movie_data.csv")

# Print the shape of the data
print("Data shape:", df.shape)

# Print top 5 datapoints
print("Top 5 datapoints:")
print(df.head(5))

# Create a new column "Category" to represent sentiment as 1 for positive and 0 for negative
df["Category"] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Check the distribution of 'Category'
print("Category distribution:")
print(df.Category.value_counts())

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.review, df.Category, test_size=0.2)

# Create a CountVectorizer
v = CountVectorizer()
x_train_cv = v.fit_transform(X_train.values)
# Create a RandomForestClassifier pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),                                                  
    ('random_forest', RandomForestClassifier(n_estimators=50, criterion='entropy'))      
])

# Fit the pipeline with training data
clf.fit(X_train, y_train)

# Get predictions for X_test
y_pred = clf.predict(X_test)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy, precision, and recall
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Plot a bar graph to show the sentiment distribution
plt.figure(figsize=(6, 4))
df.Category.value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
plt.tight_layout()
plt.show()
