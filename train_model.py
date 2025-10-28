import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_dataset():
    # ✅ Automatically find the dataset path (works anywhere)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(repo_root, "dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {csv_path}")
    print(f"📂 Loading dataset from: {csv_path}")

    return pd.read_csv(csv_path)

def bin_crime_count(count):
    if count < 30000:
        return 0  # Low
    elif count <= 40000:
        return 1  # Medium
    else:
        return 2  # High

def main():
    # ✅ Load and preview dataset
    df = load_dataset()
    print("\n📄 First few rows of dataset:")
    print(df.head())

    # ✅ Prepare data
    le = LabelEncoder()
    df['city_encoded'] = le.fit_transform(df['city'])
    X = df[['city_encoded', 'year']]
    y = df['crime_count'].apply(bin_crime_count)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Build and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Evaluate performance
    preds = model.predict(X_test)
    print("\n✅ Model Performance:")
    print("Accuracy:", round(accuracy_score(y_test, preds), 2))
    print(classification_report(y_test, preds, target_names=['Low', 'Medium', 'High']))

    # ✅ Save trained model and encoder
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "crime_model.pkl")
    encoder_path = os.path.join(out_dir, "label_encoder.pkl")
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    print(f"\n💾 Model saved successfully at: {model_path}")
    print(f"💾 Encoder saved successfully at: {encoder_path}")

    # ✅ Predict for 2025 for all cities
    cities = sorted(df["city"].unique())
    future_df = pd.DataFrame({"city": cities, "year": 2025})
    future_df['city_encoded'] = le.transform(future_df['city'])
    future_X = future_df[['city_encoded', 'year']]
    future_df["predicted_risk_2025"] = model.predict(future_X)
    future_df["predicted_risk_2025"] = future_df["predicted_risk_2025"].map({0: 'Low', 1: 'Medium', 2: 'High'})
    print("\n📊 Predictions for 2025:")
    print(future_df[['city', 'predicted_risk_2025']])

if __name__ == "__main__":
    main()
