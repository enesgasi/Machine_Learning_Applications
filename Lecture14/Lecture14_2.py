import speech_recognition as sr
import pandas as pd
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy

# -----------------------------
# 1. Feature extraction
# -----------------------------
def name_features(name, n):
    name = name.lower()
    return {
        f"first_{n}_letters": name[:n]
    }

# -----------------------------
# 2. Read CSV & label data
# -----------------------------
df = pd.read_csv("names.csv")

def label_name(name):
    return "short" if len(name) < 6 else "long"

df["label"] = df["name"].apply(label_name)

# -----------------------------
# 3. Shuffle data
# -----------------------------
data = list(zip(df["name"], df["label"]))
random.shuffle(data)

# -----------------------------
# 4. Train-test split (80-20)
# -----------------------------
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
test_data = data[split_index:]

# -----------------------------
# 5. Train classifiers
# -----------------------------
classifiers = {}

for n in [1, 2, 3]:
    train_set = [(name_features(name, n), label) for name, label in train_data]
    test_set = [(name_features(name, n), label) for name, label in test_data]

    classifier = NaiveBayesClassifier.train(train_set)
    acc = nltk_accuracy(classifier, test_set)

    classifiers[n] = classifier
    print(f"Accuracy using first {n} letter(s): {acc:.2f}")

# -----------------------------
# 6. Speech recognition
# -----------------------------
use_speech = input("Use microphone? (y/n): ").lower()

if use_speech == "y":
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say a name...")
            audio = recognizer.listen(source)


        spoken_name = recognizer.recognize_google(audio)
        print("Recognized name:", spoken_name)

    except Exception as e:
        print("Speech recognition failed, switching to text input.")
        spoken_name = input("Enter a name: ")

else:
    spoken_name = input("Enter a name: ")

    # -----------------------------
    # 7. Classification result
    # -----------------------------
    for n in [1, 2, 3]:
        result = classifiers[n].classify(name_features(spoken_name, n))
        print(f"Prediction using first {n} letter(s): {result}")


