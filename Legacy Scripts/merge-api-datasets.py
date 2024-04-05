import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Function to extract prompt-response pairs as strings, handling various data formats
def extract_prompt_response_string_corrected(record):
    try:
        response_data = json.loads(record['response'])
        prompt = str(response_data.get('prompt', ""))
        response = str(response_data.get('response', ""))
        return prompt + " " + response
    except (KeyError, json.JSONDecodeError):
        return ""

# Function to find similar entries within a limited scope
def find_similar_within_scope(tfidf_matrix, index, scope_indices, threshold=0.8):
    cosine_similarities = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix[scope_indices]).flatten()
    similar_indices = np.where(cosine_similarities > threshold)[0]
    return similar_indices

# Function to get the highest paragraph number for each book title
def get_highest_paragraph_numbers(data):
    highest_numbers = {}
    for record in data:
        book_title = record['book_title']
        paragraph_index = record['paragraph_index']
        if book_title not in highest_numbers:
            highest_numbers[book_title] = paragraph_index
        else:
            highest_numbers[book_title] = max(highest_numbers[book_title], paragraph_index)
    return highest_numbers

# Function to adjust paragraph numbers for distinct conversations
def adjust_paragraph_numbers_for_distinct_conversations(data, duplicates):
    highest_paragraph_numbers = get_highest_paragraph_numbers(data)
    for key, duplicate_records in duplicates.items():
        book_title, _ = key
        for record in duplicate_records:
            if record not in data:
                highest_paragraph_numbers[book_title] += 1
                record['paragraph_index'] = highest_paragraph_numbers[book_title]
    return data

# Function to find duplicates in the dataset
def find_duplicates(data):
    duplicates = {}
    for record in data:
        key = (record['book_title'], record['paragraph_index'])
        if key not in duplicates:
            duplicates[key] = [record]
        else:
            duplicates[key].append(record)
    # Filter out keys with only one record (not duplicates)
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
    return duplicates

# Load datasets
data1 = load_jsonl('larger_dataset.jsonl')
data_smaller = load_jsonl('smaller_dataset.jsonl')

# Extract prompt-response pairs as strings
data1_prompts_responses_str = [extract_prompt_response_string_corrected(record) for record in data1]
data_smaller_prompts_responses_str = [extract_prompt_response_string_corrected(record) for record in data_smaller]

# Vectorize the prompt-response pairs
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data1_prompts_responses_str + data_smaller_prompts_responses_str)

# Create a dictionary for the larger dataset for limited scope comparison
data1_dict = {}
for record, pr_str in zip(data1, data1_prompts_responses_str):
    key = (record['book_title'], record['paragraph_index'])
    if key not in data1_dict:
        data1_dict[key] = []
    data1_dict[key].append(pr_str)

# Identifying unique entries from the smaller dataset
unique_entries_indices = []
new_paragraph_number = max(record['paragraph_index'] for record in data1) + 1

for i, (record, entry_str) in enumerate(zip(data_smaller, data_smaller_prompts_responses_str)):
    key = (record['book_title'], record['paragraph_index'])
    if key in data1_dict:
        scope_indices = [data1_prompts_responses_str.index(pr_str) for pr_str in data1_dict[key]]
        similar_indices = find_similar_within_scope(tfidf_matrix, i + len(data1_prompts_responses_str), scope_indices)
        if len(similar_indices) == 0:
            unique_entries_indices.append(i)

# Extract and update the unique records from the smaller dataset
unique_records_smaller = [data_smaller[i] for i in unique_entries_indices]
for record in unique_records_smaller:
    record['paragraph_index'] = new_paragraph_number
    new_paragraph_number += 1

# Merge the datasets
final_merged_data = data1 + unique_records_smaller

# Find duplicates
duplicates_in_data = find_duplicates(final_merged_data)

# Adjust paragraph numbers for distinct conversations
final_data_with_adjusted_paragraphs = adjust_paragraph_numbers_for_distinct_conversations(final_merged_data, duplicates_in_data)

# Save the final dataset
with open('final_merged_dataset.jsonl', 'w') as file:
    for record in final_data_with_adjusted_paragraphs:
        file.write(json.dumps(record) + '\n')
