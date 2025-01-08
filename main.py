from app import run_streamlit_app
from train import run_training_pipeline

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Train the model
        print("Starting training pipeline...")
        model, scaler, report, conf_matrix = run_training_pipeline()
        print("\nModel Performance Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
    else:
        # Run the Streamlit app
        run_streamlit_app()