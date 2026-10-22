# AI-Based Overvoltage and Undervoltage Condition Classification

## Project overview
A classroom capstone demonstrating three-class voltage classification using Python and Random Forest, with a Streamlit dashboard. The dataset is simulated; it is not real field data.

## Illustrative project labels
For a simulated 12 V DC system:
- Voltage < 10.8 V: `Undervoltage`
- 10.8 V <= voltage <= 13.2 V: `Normal`
- Voltage > 13.2 V: `Overvoltage`

These are project demonstration thresholds only. Actual acceptable ranges depend on the real device and its specifications.

## Requirements
Python 3.10+ recommended.

Install:
```bash
pip install -r requirements.txt
```

Train model and create dataset/evaluation outputs:
```bash
python train_model.py
```

Run dashboard from the same folder:
```bash
streamlit run app.py
```

## Files
- `train_model.py`: simulated dataset, EDA, training, metrics, confusion matrix, saved model
- `app.py`: interactive dashboard
- `voltage_dataset.csv`: generated 3,000-row dataset
- `voltage_model.pkl`: trained model (created by running training script)
- `voltage_distribution.png`, `confusion_matrix.png`: generated graphs
- `evaluation_summary.txt`: generated metrics
- `Project_Report.docx`: report
- `Project_Presentation.pptx`: slides

## Important limitation
Because labels are generated directly from voltage thresholds and voltage is an input feature, excellent test accuracy is expected and does not demonstrate robust real-world fault detection. For a stronger study, use independently collected, safely obtained, labelled data, separate test sessions, and validation across different loads and devices.

## Safety
Do not connect student-built circuits directly to mains electricity. Any physical demonstration should use a suitable isolated, current-limited low-voltage supply, correctly rated sensors, and instructor supervision. This dashboard is not a certified protection device.
