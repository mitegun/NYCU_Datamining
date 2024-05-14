import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import torch
from datasets import load_metric
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Load the training data from a JSON file
train_dataframe = pd.read_json('train.json')

# Split the data into training and validation sets
train_dataframe, validation_dataframe = train_test_split(train_dataframe, test_size=0.33, random_state=42)

# Combine the "title" and "text" columns into a single column
train_dataframe['combined_text'] = train_dataframe['title'] + " " + train_dataframe['text']
validation_dataframe['combined_text'] = validation_dataframe['title'] + " " + validation_dataframe['text']

# Extract the combined texts and ratings
X_train, y_train = train_dataframe['combined_text'].to_list(), train_dataframe['rating'].astype(int).to_list()
X_val, y_val = validation_dataframe['combined_text'].to_list(), validation_dataframe['rating'].astype(int).to_list()

# Adjust the ratings to be zero-indexed (ratings go from 0 to 4 instead of 1 to 5)
y_train = [grade - 1 for grade in y_train]
y_val = [grade - 1 for grade in y_val]

# Define the mappings between IDs and class labels
id2label = {i: f"{i+1} star" for i in range(5)}
label2id = {v: k for k, v in id2label.items()}

# Load the tokenizer from the specified model
model_path = "LiYuan/amazon-review-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the preprocessing function for examples
def preprocess_function(example):
    text = example['combined_text']
    rating = example['rating'] - 1
    tokenized_example = tokenizer(text, truncation=True, padding='max_length', max_length=30)
    tokenized_example['label'] = rating
    return tokenized_example

# Apply the preprocessing function to the training and validation datasets
train_dataset = Dataset.from_pandas(train_dataframe).map(preprocess_function, remove_columns=['title', 'text'])
val_dataset = Dataset.from_pandas(validation_dataframe).map(preprocess_function, remove_columns=['title', 'text'])

# Create a DataCollatorWithPadding object
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define the function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    mae = mean_absolute_error(labels, predictions)
    return {"mean_absolute_error": mae}

# Initialize the custom model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(id2label))

# Define the training arguments
training_args = TrainingArguments(
    output_dir="my_model",
    logging_dir='logs',
    evaluation_strategy="epoch",
    num_train_epochs=11,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create a Trainer object with the training and validation datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start the training process
trainer.train()

# Load the trained tokenizer and model
model_path = '/my_model/checkpoint-32252/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the test data from a JSON file
df_test = pd.read_json('test.json')

# Combine the "title" and "text" columns into a single column
df_test['combined_text'] = df_test['title'] + " " + df_test['text']

# Preprocess the test data
def preprocess_test_data(example):
    text = example['combined_text']
    tokenized_input = tokenizer(text, truncation=True, padding='max_length', max_length=30, return_tensors='pt')
    return {
        'input_ids': tokenized_input['input_ids'].squeeze(),
        'attention_mask': tokenized_input['attention_mask'].squeeze()
    }

# Apply the preprocessing function to the test dataset
test_dataset = df_test.apply(preprocess_test_data, axis=1).tolist()
test_dataset = torch.utils.data.TensorDataset(
    torch.stack([example['input_ids'] for example in test_dataset]),
    torch.stack([example['attention_mask'] for example in test_dataset])
)

# Create a DataLoader from the test dataset
batch_size = 16
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Function to make predictions using the test dataloader, tokenizer, and model
def make_predictions(test_dataloader, tokenizer, model):
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, padding=True, truncation=True)
    predictions = []

    for batch in test_dataloader:
        decoded_texts = tokenizer.batch_decode(batch[0].tolist(), skip_special_tokens=True)
        predictions.extend(classifier(decoded_texts))

    # Extract the prediction results and format them for export
    prediction_results = extract_prediction_results(predictions)
    format_export_answers(prediction_results)

# Extract the prediction results from the model output
def extract_prediction_results(pred):
    res = []
    for item in pred:
        label = item['label']
        class_index = int(label.split()[0]) - 1
        grade = class_index + 1
        res.append(grade)
    return res

# Format the extracted answers and export them to a CSV file
def format_export_answers(answers):
    fields = ["index", "answer"]
    rows = [["index_" + str(i), answers[i]] for i in range(len(answers))]
    with open('answer.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

# Execute the prediction and export process
make_predictions(test_dataloader, tokenizer, model)
