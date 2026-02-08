with open('app/streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Using the exact text with Unicode apostrophe character
old_text = "The **confidence score** represents the model's estimated probability for the predicted category.\nIt is extracted directly from the model's internal probability output and is not a separate\nevaluation metric."

new_text = """The **confidence score** represents the model's raw confidence level for the predicted category.

⚠️ **Important Disclaimers:**
- This is a **model-dependent confidence score**, not a calibrated probability
- Different models (XGBoost, RandomForest, SVM, etc.) produce different confidence scores
- SVM and RandomForest models often have poorly-calibrated probabilities
- **Always combine this score with additional assessment methods** before making decisions
- Use this as a **relative measure of model certainty**, not an absolute probability"""

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('app/streamlit_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Success: Text replaced')
else:
    print('Error: Text not found')
