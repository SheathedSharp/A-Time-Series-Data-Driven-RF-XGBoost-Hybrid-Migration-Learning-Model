import os
import joblib

def load_pre_trained_model(need_load=True):
    if not need_load:
        return None, False

    model_folder = 'model/'
    available_models = [f for f in os.listdir(model_folder) if f.endswith('.pkl')]

    print("\n" + "="*50)
    print("Pre-trained Model Selection".center(50))
    print("="*50 + "\n")

    if not available_models:
        print("No pre-trained models found.")
        print("\n" + "="*50 + "\n")
        return None, False

    print("Available pre-trained models:")
    for i, model_file in enumerate(available_models, 1):
        print(f"{i}. {model_file}")

    print("\n" + "-"*50)

    while True:
        try:
            choice = int(input("Enter the number of the model you want to load (0 to train a new model): "))
            if choice == 0:
                print("\n" + "="*50 + "\n")
                return None, False
            if 1 <= choice <= len(available_models):
                selected_model = available_models[choice - 1]
                model_path = os.path.join(model_folder, selected_model)
                print(f"\nLoading pre-trained model from {model_path}")
                model = joblib.load(model_path)
                print("\n" + "="*50 + "\n")
                return model, True
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")