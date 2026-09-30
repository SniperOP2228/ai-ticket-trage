import os
import time
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    train_df = pd.read_csv(r"d:\IIT B\ai-ticket-triage\data\processed\train.csv").dropna(subset=['text'])
    val_df = pd.read_csv(r"d:\IIT B\ai-ticket-triage\data\processed\val.csv").dropna(subset=['text'])
    
    vectorizer = joblib.load(r"d:\IIT B\ai-ticket-triage\ml\models\tfidf_vectorizer.joblib")
    X_train = vectorizer.transform(train_df['text'])
    y_train = train_df['category']
    X_val = vectorizer.transform(val_df['text'])
    y_val = val_df['category']
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        y_pred = model.predict(X_val)
        inference_time = time.time() - t0
        
        acc = accuracy_score(y_val, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Weighted F1': f1,
            'Macro F1': macro_f1,
            'Train Time': train_time,
            'Inference Time': inference_time
        })
    
    results_df = pd.DataFrame(results)
    print("Model Comparison:")
    print(results_df)
    
    os.makedirs(r"d:\IIT B\ai-ticket-triage\ml\evaluation", exist_ok=True)
    results_df.to_csv(r"d:\IIT B\ai-ticket-triage\ml\evaluation\category_model_comparison.csv", index=False)

if __name__ == "__main__":
    main()
