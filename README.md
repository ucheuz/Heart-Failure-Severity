# Heart-Failure-Severity
A machine learning algorithm to diagnose severity of heart-failure, by distinguishing between  healthy patients (Label 0), patients with mild heart failure (Label 1), patients with severe heart failure (Label 2). The project includes:
  A Python Script (main.py) that runs the model.
  A Juypter Notebook (Heart Failure Severity Juypter Notebook.ipynb) Task 2 with exploratory data analyis, model training and conclusions about the model at each stage.
  A Tableau visualisation of the actual and predicted data.

# Description
In this project, I aimed to develop a machine learning model to diagnose the severity of heart failure by classifying patients into three categories: healthy, mild heart failure, and severe heart failure. The model utilised ejection fraction (EF) and QRS indices as key features to distinguish between these conditions. To improve classification performance, I applied a non-linear multinomial logistic regression with polynomial feature transformation, ensuring the model could capture complex patterns in the data. By using L2 regularisation to prevent overfitting and class balancing to handle underrepresented classes, the model achieved a high accuracy of 88%, with strong recall scores across all categories. The final implementation included grid search tuning, visualisation of the decision boundary and confusion matrix, and an evaluation of recall values to ensure effective diagnosis across all severity levels.

# How to run
Option 1: Running the Python Script: Can be done directly in terminal. This will train the model and output predictions and accuracy scores.

Option 2: Running the Juypter Notebook: Download and open Heart Failure Severity Juypter Notebook.ipynb and run the cells in Task 2 step by step.

# Dataset
File: datasets/heart_failure_data_complete.csv
Description: A collection of patient records with features like HF (heart failure) labe, Ejection Fraction (EF), Global Longitudinal Strain (GLS) and QRS Complex (QRS)

# Results
Accuracy: 88%.
Tableau Visualisation: "https://public.tableau.com/views/HeartFailureSeverityML/Dashboard?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
Conclusion: The performance numbers for the final model on the test data indicated a really good predictive capability. The accuracy is a value of 0.88: This means that the model correctly classified 88% of the test samples, which is a high result for a multi-class classification problem. The recall values are [0.92, 0.90, 1.0]: The recalls for each class showed that the model is highly effective at identifying samples across all heart failure classes: For Healthy (Class 0): A recall of 0.92 indicates that 92% of the actual healthy samples were correctly identified. For Mild HF (Class 1): A recall of 0.90 means that 90% of the mild heart failure cases were detected. For Severe HF (Class 2): A perfect recall of 1.0 suggests that all severe heart failure cases were correctly classified. The high average recall value of 0.94 demonstrates that the model maintains strong performance across all classes, suggesting it effectively balances precision and recall without being biased toward any single class. The model appears to generalize well to the test data. This is evident from the close alignment between the training and test performance numbers.

# Contributors
Kosiasochukwu Uchemudi Uzoka - Author
