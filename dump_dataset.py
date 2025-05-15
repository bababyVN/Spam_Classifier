import pandas as pd
import os
from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))

# read dataset UCI SMS Spam Collection
uci_data_path = os.path.join(script_dir, 'Data\SMSSpamCollection')
uci_data = pd.read_csv(uci_data_path, sep = '\t', names = ['label', 'text'])
uci_data['label'] = uci_data['label'].map({'spam': 1, 'ham': 0})

# read dataset Telegram Spam or Ham from Kaggle
telegram_data_path = os.path.join(script_dir, 'Data\\telegram_spam.csv')
telegram_data = pd.read_csv(telegram_data_path)
telegram_data['label'] = telegram_data['text_type'].map({'spam': 1, 'ham': 0})  

# read dataset Enron-Spam from Hugging Face
enron_dataset = load_dataset('SetFit/enron_spam')
enron_data = pd.DataFrame(enron_dataset['train'])
enron_data = enron_data[['text', 'label']]

# combine and shuffle datasets
combined_data = pd.concat([uci_data, telegram_data, enron_data], ignore_index=True)
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# dump combined_data to a CSV file
output_path = os.path.join(script_dir, 'Data\combined_spam_dataset.csv')
combined_data.to_csv(output_path, index=False)
print(f"Combined dataset saved to: {output_path}")
