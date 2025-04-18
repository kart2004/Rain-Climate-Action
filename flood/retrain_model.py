# Change the import to use relative path since we're in the same package
from flood.load import retrain_model

if __name__ == "__main__":
    print("Starting model retraining...")
    model = retrain_model()
    print("Model retraining completed successfully!")