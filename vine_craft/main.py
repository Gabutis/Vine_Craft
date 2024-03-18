import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def add_gaussian_noise(X, noise_level=0.3):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    return X + noise

data = load_wine()
X, y = data.data, data.target

X_noisy = add_gaussian_noise(X, noise_level=0.3)

X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

pca_pipe = Pipeline([
    ('pca', PCA(n_components=2)),
    ('logistic', LogisticRegression(max_iter=1000))
])
pca_param_grid = {
    'logistic__C': [0.1, 1, 10],
    'logistic__solver': ['liblinear', 'saga']
}
pca_grid_search = GridSearchCV(pca_pipe, pca_param_grid, cv=5, scoring='accuracy')
pca_grid_search.fit(X_train_scaled, y_train)

print("Best parameters with PCA:", pca_grid_search.best_params_)
print("Best cross-validation score with PCA: {:.2f}".format(pca_grid_search.best_score_))

final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {test_accuracy}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y), rotation=45)
plt.yticks(tick_marks, np.unique(y))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()
