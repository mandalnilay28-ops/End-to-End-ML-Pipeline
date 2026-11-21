
from sklearn.datasets import fetch_openml
from model_trainer import ModelTrainer

# Load the Ames Housing Dataset
print("Loading dataset...")

ames = fetch_openml(data_id=42165, as_frame=True)

data = ames.data.copy()
data["SalePrice"] = ames.target

# Remove ID column if present
if "Id" in data.columns:
    data = data.drop(columns=["Id"])

print("Dataset loaded successfully!")

# Initialize ModelTrainer
trainer = ModelTrainer(data)

# Execute complete pipeline
trainer.clean_data()
trainer.prepare_data()
trainer.train_model()

# Evaluate model
results = trainer.evaluate_model()

# Visualize predictions
trainer.plot_results()

# Predict a house price
new_house = trainer.X_test.iloc[[0]]
trainer.predict(new_house)

# Save trained model
trainer.save_model("house_price_model.pkl")

print("\nProject execution completed!")
