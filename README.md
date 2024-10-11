# Deepcraft-Trainee-stock-predication-model


## **Overview**
This project builds a stock price prediction model using an LSTM (Long Short-Term Memory) network. The data used is NTT stock price data, and the goal is to predict future stock prices based on historical trends. The steps include:

1. **Exploratory Data Analysis (EDA):** Visualize stock price trends, check correlations, and detect any missing values.
2. **Data Preprocessing and Feature Engineering:** Normalize the data, handle missing values, and generate technical indicators like rolling averages.
3. **Model Selection and Training:** Build an LSTM model to predict stock prices using time-series data.
4. **Model Evaluation:** Assess the accuracy of the model using RMSE (Root Mean Squared Error) and plot actual vs predicted prices.
5. **Model Improvement:** Refine the model based on results, e.g., by adding more LSTM layers or adjusting hyperparameters.

The model uses Keras for deep learning and Matplotlib for visualization. The predictions are evaluated using a test set derived from the provided stock price data.

## **Requirements**

### **Libraries and Dependencies**
To run this program, you need to install the following libraries:

1. **Python 3.x**
2. **Pandas** (For data manipulation)
    ```bash
    pip install pandas
    ```
3. **NumPy** (For numerical operations)
    ```bash
    pip install numpy
    ```
4. **Matplotlib** (For plotting graphs)
    ```bash
    pip install matplotlib
    ```
5. **Seaborn** (For enhanced data visualizations)
    ```bash
    pip install seaborn
    ```
6. **TensorFlow** (For building and training the LSTM model)
    ```bash
    pip install tensorflow
    ```
7. **Scikit-Learn** (For scaling and evaluation metrics)
    ```bash
    pip install scikit-learn
    ```

### **How to Install Dependencies**
You can install the necessary dependencies by running the following command:

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
```

## **Steps to Run the Program**

### **1. Clone the Project**
Clone the project to your local machine or ensure you have the script and the dataset (CSV file) in the same folder.

### **2. Dataset**
Ensure the stock price dataset (CSV) is in the same directory as the script or update the file path in the code.

### **3. Running the Program**
Once dependencies are installed, follow these steps:

1. **Download the Dataset:**
   Ensure you have the dataset file (in CSV format) ready and correctly placed.

2. **Run the Python Script:**
   Execute the script by running the following command in your terminal or IDE:

   ```bash
   python stock_price_prediction.py
   ```

3. **Output:**
   The script will generate:
   - Graphs of the stock price trends over time.
   - Visualization of actual vs predicted stock prices.
   - The Root Mean Squared Error (RMSE) metric to evaluate model performance.

### **4. Model Improvement**
If the results need further optimization, you can retrain the model by increasing the complexity (more LSTM layers, changing the batch size, or training for more epochs). You can find the retraining steps under the "Model Improvement" section in the script.

## **Project Structure**

- **`stock_price.csv`**: The stock price dataset (ensure the file contains columns like `Date` and `Close`).
- **`stock_price_prediction.py`**: The main Python script containing all steps from EDA, model building, and evaluation.
- **`README.md`**: This README file.

## **File Details**
- **`stock_price_prediction.py`**: 
  - Loads the dataset.
  - Performs data exploration and visualization.
  - Preprocesses data and creates new features.
  - Builds and trains an LSTM model for stock price prediction.
  - Evaluates the model using RMSE and plots the results.

## **Result Evaluation**

The script will output:
- A **graph** showing the predicted stock prices compared to the actual prices.
- The **RMSE** value, which quantifies the model's prediction accuracy.

### **Modifying the Code**
- **Change Dataset**: If you want to change the dataset, update the path to the CSV file in the `read_csv()` function.
- **Tune Hyperparameters**: You can modify LSTM hyperparameters (like `epochs`, `batch_size`, `units`) for better results.
- **Add Features**: You can add more features like external economic factors or other technical indicators for a more sophisticated model.

---
