import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from analyze_sentiment import predict_single_sentiment


def main():
    # 1️⃣ Dataseti yükle
    df = pd.read_csv("../turkish_balanced_eval_set.csv")

    print(f"Loaded: {len(df)} samples")

    # 2️⃣ Model tahmini
    df["predicted_label"], df["predicted_score"] = zip(
        *df["text"].apply(predict_single_sentiment)
    )

    # 3️⃣ Accuracy
    acc = accuracy_score(df["manual_label"], df["predicted_label"])
    print(f"\n🎯 Accuracy: {acc:.4f}")

    # 4️⃣ Detailed report
    print("\n📊 Classification Report:")
    print(classification_report(df["manual_label"], df["predicted_label"]))

    # 5️⃣ Confusion matrix
    print("\n🔢 Confusion Matrix:")
    print(confusion_matrix(df["manual_label"], df["predicted_label"]))

    # 6️⃣ Hatalar
    errors = df[df["manual_label"] != df["predicted_label"]]

    print(f"\n❌ Total Errors: {len(errors)}")

    print("\n--- Sample Errors ---")
    for _, row in errors.head(10).iterrows():
        print(f"\nTEXT: {row['text']}")
        print(f"TRUE: {row['manual_label']}")
        print(f"PRED: {row['predicted_label']}")

    # 7️⃣ Kaydet
    df.to_csv("../evaluation_results.csv", index=False)
    print("\n✅ Results saved to evaluation_results.csv")


if __name__ == "__main__":
    main()