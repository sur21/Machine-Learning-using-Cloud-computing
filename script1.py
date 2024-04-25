import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

def train_model(n_estimators, random_state):
    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=random_state)
    
    # Train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    
    # Save model
    joblib.dump(clf, 'model.joblib')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RandomForestClassifier')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
    parser.add_argument('--random_state', type=int, default=0, help='Seed used by the random number generator')
    parser.add_argument("--train-file", type=str, default="train-v-1.csv")
    parser.add_argument("--test-file", type=str, default="test-v-1.csv")
    args = parser.parse_args()
    
    try:
        train_model(args.n_estimators, args.random_state)
        print("Training job completed successfully!")
    except Exception as e:
        print(f"Training job failed with error: {str(e)}")











