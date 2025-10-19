import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from matplotlib import pyplot as plt

# --- 1. Load Model and Data ---
# It's better practice to load data into a separate variable for clarity
model = joblib.load("./artifacts/model.joblib")
data = pd.read_csv("./data.csv")

# --- 2. Prepare Features and Target ---
# Assuming 'species' is the target column
X = data.drop("species", axis=1)
y = data["species"]

# --- 3. Make Predictions ---
y_pred = model.predict(X)


"""# Generate the report as a dictionary
report = classification_report(y, y_pred, output_dict=True)

# Convert the dictionary to a DataFrame for easy CSV saving
df = pd.DataFrame(report).transpose()

# Round the metric values for cleaner output in the CSV/Markdown table
df = df.round(4) 

# Save the metrics to the required path for the CSV-to-MD GitHub Action
df.to_csv("./metrics/report.csv", index_label='Metric')"""

# 1. Generate the report as a string directly
report_string = classification_report(y, y_pred)

# 2. Save the string report to a text file
with open("./artifact_metrics/classification_report.txt", "w") as f:
    f.write(report_string)


# Create the confusion matrix array
cm = confusion_matrix(y, y_pred)

# Initialize the display object
cmd = ConfusionMatrixDisplay(cm, display_labels=y.unique())

# Set up the plot for cleaner saving
fig, ax = plt.subplots(figsize=(8, 6)) # Define figure size
cmd.plot(ax=ax, cmap=plt.cm.Blues)    # Plot on the defined axes
plt.title("Model Confusion Matrix")   # Add a title

# Save the plot to the required path, explicitly using the .png extension
# Use bbox_inches='tight' to prevent labels from being cut off
plt.savefig("./artifact_metrics/confusion_matrix.png", bbox_inches='tight')

# --- 6. Print DataFrame (for immediate console feedback/debugging) ---
print("\nMetrics Report:")
print(df)