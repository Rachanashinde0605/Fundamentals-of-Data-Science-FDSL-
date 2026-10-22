import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv('Titanic_Dataset.csv')

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

# 2. Line Plot - Fare trend by Passenger ID
plt.figure(figsize=(8, 5))
plt.plot(df['PassengerId'], df['Fare'], marker='o')
plt.title('Fare Trend by Passenger ID')
plt.xlabel('Passenger ID')
plt.ylabel('Fare')
plt.show()


# 3. Bar Plot - Average Fare by Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Fare', data=df)
plt.title('Average Fare by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Average Fare')
plt.show()


# 4. Histogram - Distribution of Passenger Age
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'].dropna(), bins=10, kde=True)
plt.title('Distribution of Passenger Age')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()


# 5. Scatter Plot - Age vs Fare
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Fare', hue='Sex', data=df)
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


# 6. Box Plot - Fare Distribution by Passenger Class
plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()


# 7. Pie Chart - Gender-wise Passenger Count
gender_counts = df['Sex'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title('Gender-wise Distribution of Passengers')
plt.show()


# 8. Heatmap - Correlation between Numeric Variables
plt.figure(figsize=(8, 6))

corr = df[
    ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title('Correlation Heatmap')
plt.show()