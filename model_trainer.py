class ModelTrainer:

    # 1. Constructor
    def __init__(self, data):
        self.data = data.copy()
        self.model = None

        print("ModelTrainer object created successfully!")
        print("Dataset shape:", self.data.shape)

    # 2. Data cleaning
    def clean_data(self):

        self.data = self.data.drop_duplicates()

        print("\nData cleaning completed!")
        print("Dataset shape:", self.data.shape)

    # 3. Data preparation
    def prepare_data(self):

        X = self.data.drop("SalePrice", axis=1)
        y = self.data["SalePrice"]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

        print("\nData preparation completed!")
        print("Training records:", len(self.X_train))
        print("Testing records:", len(self.X_test))

    # 4. Model training
    def train_model(self):

        numerical_cols = self.X_train.select_dtypes(
            include=["number"]
        ).columns

        categorical_cols = self.X_train.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        # Numerical preprocessing
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        # Categorical preprocessing
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine preprocessing methods
        preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        # Complete machine learning pipeline
        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

        # Train model
        self.model.fit(self.X_train, self.y_train)

        print("\nLinear Regression model trained successfully!")

    # 5. Model evaluation
    def evaluate_model(self):

        predictions = self.model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, predictions)

        print("\n----- MODEL EVALUATION -----")
        print("Mean Squared Error:", mse)
        print("Root Mean Squared Error:", rmse)
        print("R2 Score:", r2)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }

    # 6. Visualization
    def plot_results(self):

        predictions = self.model.predict(self.X_test)

        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.y_test,
            predictions,
            alpha=0.6
        )

        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--"
        )

        plt.xlabel("Actual House Prices")
        plt.ylabel("Predicted House Prices")
        plt.title("Actual vs Predicted House Prices")

        plt.show()

    # 7. Prediction
    def predict(self, new_data):

        predictions = self.model.predict(new_data)

        print("\n----- HOUSE PRICE PREDICTION -----")

        for i, price in enumerate(predictions):
            print(f"House {i+1}: ${price:,.2f}")

        return predictions

    # 8. Save trained model
    def save_model(self, filename="house_price_model.pkl"):

        joblib.dump(self.model, filename)

        print("\nModel saved successfully!")
        print("File:", filename)
