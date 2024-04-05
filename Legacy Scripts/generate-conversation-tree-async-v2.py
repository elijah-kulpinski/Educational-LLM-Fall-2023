"""
File: generate-conversation-tree-async-v2.py
Author: Elijah Kulpinski
Date: 11/18/23
Version: 2.1.1

Description:
    This script, updated to version 2.1.0, refines the conversation tree generation process for educational dialogues.
    Enhancements include deduplication logic, improved parsing and error handling, detailed quality check logging, 
    and various other refinements to ensure high-quality, diverse datasets for LLM fine-tuning.

Key Features:
    - Asynchronous Processing: Manages concurrent tasks to efficiently generate conversation trees.
    - Deduplication: Prevents the creation of duplicate conversation entries.
    - Enhanced Parsing: Robust handling of API responses to ensure valid JSON formatting.
    - Detailed Quality Check Feedback: Provides specific reasons for quality check failures to aid in debugging.
    - Improved Backoff Mechanism: Strategically retries after recoverable errors, including rate limits.
    - Finer-Grained Error Handling: Allows precise tracking of issues at the paragraph level.
    - Enhanced Logging: Offers comprehensive logging for better traceability and debugging.

Version History:
    - 2.1.1: Moved conversation quality check to using OpenAI's API for consistency, quality, and performance.
    - 2.1.0: Added deduplication, enhanced parsing logic, more advanced quality check NLP checks, and improved error handling.
    - 2.0.0: Introduced a granular rating system, sophisticated rehaul of the branching logic, automated data quality checks, and cost-efficient API utilization strategies.
    - 1.3.0: Implemented log file generation and console output for improved monitoring and debugging. Conducted testing with 'output-snippet.xml' and processed the resulting 'conversation_trees.jsonl' file for data integrity.
    - 1.2.0: Introduced dual-branching logic to simulate 'good' and 'bad' educational responses, creating a dynamic dataset with varied teaching methodologies.
    - 1.1.0: Transitioned from synchronous to asynchronous processing, allowing for parallel generation of conversation trees and optimizing resource usage. Time tracking added to monitor task times for testing.
    - 1.0.0: Initial release. Generated simple, linear conversation trees in a synchronous manner. Basic API rate limit management and conversation context derived directly from textbook content.

Usage:
    Run with Python 3.x, ensuring all dependencies from 'requirements.txt' are installed and API keys are set in '.env'.
    The script processes paragraphs from an XML file and outputs conversation trees to a '.jsonl' file.

    To execute the script:
    `python.exe generate-conversation-tree-async-v2.py`

Note:
    Requires OpenAI API access. Adhere to the rate limits and usage guidelines. Ensure appropriate permissions and API quota are in place for using this script.

License: MIT License (See 'LICENSE' for Details)
"""

import asyncio
import json
import os
import re
import random
import time
import hashlib
import xml.etree.ElementTree as ET
from openai import AsyncOpenAI

# import spacy
# from textblob import TextBlob

# Constants
API_KEY = "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" # TODO: Extract from environment variable
MAX_DEPTH = 5
MAX_BRANCHES = 16
MAX_ATTEMPTS = 5
MAX_DELAY = 60
MAX_CONCURRENT_TASKS = 20
INDIVIDUAL_PAIRS_FILE = 'individual_pairs.jsonl'
RATED_TREES_FILE = 'rated_trees.jsonl'
XML_FILE_PATH = 'output_short_snippet.xml'
CONSOLE_LOG_FILE_PATH = 'console_log.txt'

# Load the spaCy model for NLP tasks
# nlp = spacy.load("en_core_web_sm") # Smaller model for testing, reduced NLP accuracy
# nlp = spacy.load("en_core_web_lg") # Larger model for deployment, improved NLP accuracy

# Ensure output and log files exist, otherwise create them
if not os.path.exists(INDIVIDUAL_PAIRS_FILE):
    open(INDIVIDUAL_PAIRS_FILE, 'w').close()

if not os.path.exists(RATED_TREES_FILE):
    open(RATED_TREES_FILE, 'w').close()

if not os.path.exists(CONSOLE_LOG_FILE_PATH):
    open(CONSOLE_LOG_FILE_PATH, 'w').close()

# Utility function to log messages to both console and text file
def display_and_log(message):
    print(message)
    with open(CONSOLE_LOG_FILE_PATH, 'a', encoding='utf-8') as log_file:
        log_file.write(f"{message}\n")

# Utility function to parse the XML file and extract paragraphs
def parse_xml(file_path):
    book_paragraphs = []
    try:
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(file_path, parser=parser)
        root = tree.getroot()
        for book in root.findall('.//book'):
            book_title = book.get('name')
            for paragraph in book.find('.//content').findall('paragraph'):
                text = paragraph.text if paragraph.text is not None else ''
                book_paragraphs.append((book_title, text.strip()))
    except Exception as e:
        display_and_log(f"Error parsing XML file at {file_path}: {e}")
    return book_paragraphs

# Utility function to append data to a file
def append_to_file(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')

# Backoff handler in case of rate limit error
async def backoff_hdlr(attempt):
    jitter = random.uniform(0, attempt)
    new_delay = min(2 ** attempt, MAX_DELAY) + jitter
    display_and_log(f"Rate limit hit, backing off for {new_delay} seconds.")
    await asyncio.sleep(new_delay) # Puts a single task to sleep not the entire program

# Function to generate a rating by querying the API to evaluate and compare conversations within a tree
async def generate_rating(conversation_tree, client):
    num_conversations = len(conversation_tree["conversation"])
    # Generate a unique identifier for each conversation to include in the prompt
    conversation_identifiers = [f"Conversation {i+1}" for i in range(num_conversations)]

    # Define the prompt to send to the API for rating the conversation
    rating_prompt = (f"Review the following conversations between a student and an educator. "
                   f"Compare each conversation against the others and rank them from 1 (most effective) to "
                   f"{num_conversations} (least effective) in terms of educational quality, considering factors "
                   f"such as clarity, engagement, and the promotion of critical thinking. "
                   f"Respond with your rankings in a list format, preceded by the conversation identifier, "
                   f"For Example: \"Conversation 1: 3, Conversation 2: 1, ..., Conversation {num_conversations}: 2\"."
    )

    # Append a serialized version of each conversation with its identifier
    for idx, identifier in enumerate(conversation_identifiers):
        conversation = json.dumps(conversation_tree["conversation"][idx], ensure_ascii=False)
        rating_prompt += f"\n\n{identifier}: {conversation}"

    # Make the API
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": rating_prompt}]
        )

        # Parse the API response to extract the rankings
        display_and_log(f"Rankings API Response: {response}")
        rankings_response = response.choices[0].message.content.text.strip().split(',')
        display_and_log(f"Rankings Contents: {rankings_response}")
        rankings = {}
        for ranking_pair in rankings_response:
            identifier, rank = ranking_pair.split(':')
            # Extract the number from the identifier (e.g., "Conversation 1" -> 1)
            conv_num = int(identifier.strip().split(' ')[1])
            rankings[conv_num] = int(rank.strip())
            display_and_log(f"Processed Ranking: {conv_num} -> {rank}")

        # Assign the rankings back to the conversation tree using the identifiers
        for index in range(num_conversations):
            conversation_tree["conversation"][index]["ranking"] = rankings[index + 1]
            display_and_log(f"Assigned Ranking to Conversation {index + 1}: {rankings[index + 1]}")

    except Exception as e:
        # Log error and handle it appropriately
        display_and_log(f"Error during rating generation: {e}")

    return conversation_tree  # Return the conversation tree with the rankings

# Function to create a conversation prompt including context management and rating instructions
async def create_conversation_prompt(book_title, paragraph, conversation_history, depth, client):
    summarized_context = await summarize_conversation(conversation_history, client) if conversation_history else ""

    # Conversation start. Root Node.
    if depth == 0:
        prompt = (
        f"Assume the role of an educator assisting a student. You are here to help them develop critical thinking skills and engage with the content."
        f"Based on the concept in the provided paragraph from the book titled: '{book_title}', "
        f"generate a prompt-response pair in JSON for use in a conversation tree in a dataset to fine-tune an LLM to be an educational assistant that "
        f"promote student critical thinking and engagement. Begin from the student's point-of-view with a question reflecting a student's inquiry or "
        f"assigned task regarding the textbook paragraph contents, the student may choose to provide part of an exerpt of the assigned paragraph or try to describe "
        f"the paragraph in their own words as context for the LLM before asking their question, students also don't always use perfect grammar, then follow up from the educator's point-of-view with a response that encourages critical thinking "
        f"and that engages the student. Your response should offer guidance, examples, or suggestions, helping the student explore the concept "
        f"further, while maintaining a professional tone. Avoid directly solving the problem; instead, focus on guiding the student to find the "
        f"solution themselves. If the conversation reaches a natural conclusion or if the topic changes significantly, include "
        f"'[CONVERSATION_END]' or '[TOPIC_END]' respectively.\nHere's the textbook paragraph that was assigned to the student: {paragraph[:500]}"
        f"Don't forget you are to generate both a 'Prompt' and a 'Response' in JSON to represent the start of the conversation between the student and educator."
        f"Example: You receive an assigned textbook paragraph from a math textbook on solving basic integrals for calculus 1 students. "
        f"Example Output: \"Prompt\": \"I just learned that integrals are the opposite of derivatives. Complete this integral: 2x^2 + 2.\", \"Response\": \"I can't solve it for you but I can help guide you towards the solution"
        f"in solving it for yourself. How many steps have you gotten in solving the integral so far? Since integrals are the opposite of derivatives, what are the steps for solving a derivative? "
        f"(Hint: Start by trying to break up this one big integral into multiple smaller, easier-to-solve integrals)\". "
        )

    # Conversation ending. Child nodes.         
    elif depth >= MAX_DEPTH - 1 and depth < MAX_DEPTH + 1:
        prompt = (
        f"Assume the role of an educator assisting a student, and are currently in conversation with the student. "
        f"Based on the concept in the provided paragraph from the book titled: '{book_title}', your goal is to start concluding the "
        f"conversation effectively without directly completing the students assignment so they can develop critical thinking skills "
        f"and engage with their course content. Your response should offer guidance, examples, or suggestions, helping the student "
        f"explore the concept further, while maintaining a professional tone. Avoid directly solving the problem; instead, focus on "
        f"guiding the student to find the solution themselves. Your job is to craft a prompt-response pair in JSON, aiming to gracefully "
        f"conclude the interaction. Ensure your response helps in summarizing key points or providing final thoughts, without introducing "
        f"new complex topics. Include '[CONVERSATION_END]' or '[TOPIC_END]' if appropriate. "
        f"\nAssigned Textbook Paragraph Contents: {paragraph[:500]}\nPrior Conversation Summary: {summarized_context}\n"
        f"Don't forget you are to generate both a Prompt and a Response in JSON to represent the conversation approaching its conclusion between the student and educator."
        f"Example 1: \"Prompt\": \"Wow, thanks! I'm feeling more confident for my exam!', \"Response\": \"You're welcome, the key is breaking down complex integrals into simpler parts. Always feel free to revisit our discussion on this topic. Good luck on your exam! [CONVERSATION_END]\""
        f"Example 2: \"Prompt\": \"I'm pretty confident on integrals, time to start English homework, can you assist with that?, \"Response\": \"I'm glad, if you ever want more practice on integrals just ask away. Let me know if I can guide you for any other school content. [TOPIC_END]\""
        )

    # Conversation middle. Sub parent nodes.             
    else:   
        prompt = (
        f"Assume the role of an educator assisting a student, and are currently in conversation with the student. "
        f"Based on the concept in the provided paragraph from the book titled: '{book_title}', your goal is to continue guiding the student towards learning "
        f"the content they need to answer their question in their inital prompt; without directly completing the students assignment outright, in order for "
        f"them to develop critical thinking skills and engage with their course content. Your response should offer guidance, examples, "
        f"or suggestions, helping the student explore the concept further, while maintaining a professional tone. Avoid directly solving "
        f"the problem; instead, focus on guiding the student to find the solution themselves. Your job is to craft a prompt-response pair "
        f"in JSON. If the conversation reaches a natural conclusion or if the topic changes significantly, include '[CONVERSATION_END]' or '[TOPIC_END]' respectively. "
        f"Assigned Textbook Paragraph Content: {paragraph[:500]}\nPrior Conversation Summary: {summarized_context}\n"
        f"Don't forget you are to generate both a Prompt and a Response in JSON to represent the progression of the conversation between the student and educator."
        f"Example 1: \"Prompt\": \"I never really understood how variables work in programming?\", \"Response\": \"Think of variables as containers for storing data. What data types do you think can be stored in these containers?\""
        f"Example 2: \"Prompt\": \"I don't really know the difference between Comparator and Comparable in Java?\", \"Response\": \"To understand the differences between Comparator and Comparable in Java, consider the purpose and "
        f"usage of each: Comparable is for natural ordering within a class (using compareTo), whereas Comparator allows for custom ordering outside a class (using compare). Reflect on their design implications, "
        f"flexibility, and specific use cases to grasp why and when each interface is used in Java programming.\""
        )
    return prompt

async def summarize_conversation(conversation_history, client):
    # Combine the conversation history into a single string.
    full_history = " ".join(conversation_history)
    
    # Create the prompt for summarization.
    summary_prompt = (
            f"Please provide a concise summary of the educational interaction below, focusing on key points and main ideas. "
            f"Ensure the summary is coherent and captures the essence of the dialogue without unnecessary details. "
            f"Example Summary: 'The student struggles with the concept of recursion. The educator guides the student by "
            f"comparing recursion to a loop and suggestes practice problems to reinforce learning including The Tower of Hanoi.'"
            f"This summary wil be used as context for prompting a large language model so make sure it's concise while comprehensive "
            f"enough for the LLM to understand the prior conversation between the student and educator.\n\n"
            f"Full Conversation: {full_history}"
        )
    
    # Make the API call for summarization.
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": summary_prompt}]
        )
        # Assuming the API returns a text response with the summary.
        display_and_log(f"API Summary Response: {response}.")
        summary = response.choices[0].message.content
        display_and_log(f"API Summary: {summary}")
        return summary
    except Exception as e:
        # Handle exceptions and possibly perform a retry or log the error.
        display_and_log(f"Error during API call for summarization: {e}")
        return ""  # Return an empty string as a fallback in case of an error.

def get_value_by_key_variants(dictionary, key_variants_dict, default=None):
    """
    Returns the value for the first key variant found in the dictionary.
    """
    result = {}
    for key, variants in key_variants_dict.items():
        for variant in variants:
            if variant in dictionary:
                result[key] = dictionary[variant]
                break
        if key not in result:
            result[key] = default
    return result

def complete_json_string(json_string, expected_structure):
    try:
        # Attempt to parse the JSON string
        json_obj = json.loads(json_string)

        # Check if the JSON object contains the expected keys
        for key, variants in expected_structure.items():
            if not any(variant in json_obj for variant in variants):
                raise json.JSONDecodeError(f"Missing '{key}' key", json_string, 0)

        return json.dumps(json_obj)

    except json.JSONDecodeError as e:
        try:
            # Check for invalid control characters and remove them
            display_and_log(f"JSONDecodeError: {e}")
            json_string = ''.join(char for char in json_string if (0x20 <= ord(char) <= 0x10FFFF) and char not in ('"', '\\', '\x08', '\x0c', '\x0a', '\x0d', '\x09', '\x00'))
            
            # Iterate through expected_structure to format JSON
            formatted_json_string = '{'
            for key, variants in expected_structure.items():
                for variant in variants:
                    # Use regex to find each variant in the JSON string
                    match = re.search(fr'"{variant}":"(.*?)"', json_string)
                    if match:
                        formatted_json_string += f'"{key}": "{match.group(1)}", '

            # Remove the trailing comma and space, if any
            if formatted_json_string.endswith(', '):
                formatted_json_string = formatted_json_string[:-2]

            formatted_json_string += '}'

            if formatted_json_string == '{}':  # If no matches were found
                raise json.JSONDecodeError("Unable to format the string correctly", json_string, 0)

            return formatted_json_string

        except json.JSONDecodeError:
            # If it's still not valid JSON, wrap it in an object
            display_and_log(f"JSONDecodeError: {e}")
            return f'{{"data": "{json_string}"}}'

def is_valid_json(json_string):
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

def parse_response(response):
    try:
        # Defining the different potential API response types 
        expected_structure = {
            "Prompt": ['Prompt', 'prompt', 'PROMPT', 'student', 'student_question', 'question'],
            "Response": ['Response', 'response', 'RESPONSE', 'educator', 'educator_response']
        }

        # Extracting the content of the assistant's message from the response
        display_and_log(f"API Response for Parsing: {response}")
        api_message = response.choices[0].message.content
        display_and_log(f"API Message Extracted: {api_message}")

        # Complete the JSON string if necessary
        if is_valid_json(api_message):
            complete_api_message = api_message
        else:
            display_and_log("Invalid JSON format")
            complete_api_message = complete_json_string(api_message, expected_structure)
            display_and_log(f"Completed JSON API Message: {complete_api_message}")

        # Parse the JSON string
        conversation = json.loads(complete_api_message)

        # Extracting 'Prompt' and 'Response', considering various capitalizations and variants
        parsed_values = get_value_by_key_variants(conversation, expected_structure, default="FAILED_TO_PARSE")
        student_question = parsed_values.get("Prompt", "FAILED_TO_PARSE")
        educator_response = parsed_values.get("Response", "FAILED_TO_PARSE")

        if student_question == "FAILED_TO_PARSE" or educator_response == "FAILED_TO_PARSE":
            raise ValueError("Failed to parse the conversation from the response.")
        else:
            display_and_log(f"Student Question: {student_question} and Educator Response: {educator_response}")
        
        return student_question, educator_response
    except Exception as e:
        display_and_log(f"Error parsing response: {e}\n")
        return "FAILED_TO_PARSE", "FAILED_TO_PARSE"

# # Function to perform automated checks on the generated conversation locally
# def perform_quality_checks_local(conversation_entry):
#     # Extract educator response and perform NLP analysis
#     response = conversation_entry['educator_response']
#     doc = nlp(response)

#     # Check for educational relevance: Presence of explanations, definitions, or clarifications
#     educational_relevance = any(token.pos_ in ["VERB", "NOUN"] and token.dep_ in ["ROOT", "attr", "dobj"] for token in doc)

#     # Sentiment analysis: Check for neutral to positive tone, as educational content should be informative and encouraging
#     sentiment = TextBlob(response).sentiment.polarity
#     positive_sentiment = sentiment >= -0.1  # Allowing slightly negative sentiment for challenging topics

#     # Contextual understanding: Analyze for cohesive and meaningful sentences
#     contextual_understanding = any(sent.root.pos_ == "VERB" and sent.root.dep_ == "ROOT" for sent in doc.sents)

#     # Length and grammar check: Assuming longer and grammatically correct responses may have more explanations/details
#     length_check = len(response.split()) > 20  # Arbitrary length threshold
#     grammar_check = any(punctuation in response for punctuation in ['.', '?', '!'])

#     # All checks must pass for the conversation entry to be considered high quality
#     quality_checks_passed = all([
#         educational_relevance,
#         positive_sentiment,
#         contextual_understanding,
#         length_check,
#         grammar_check
#     ])

#     return quality_checks_passed

# Function to perform automated checks on the generated conversation using OpenAIs API
async def perform_quality_checks_api(conversation_entry, client):
    # Create a prompt for the API to evaluate the conversation entry
    evaluation_prompt = (
        f"Please evaluate the educational quality of the following conversation "
        f"between a student and an educator. Assess it for clarity, engagement, "
        f"and promotion of critical thinking. Respond with an overall score from "
        f"1 (no conversation/unintelligible) to 50 (unhelpful/incorrect) to 100 "
        f"(perfect interaction) in JSON format with the label \'Score\'. If you "
        f"fail to respond in the expected JSON (\"Score\":\"#\"), the assumed score is 0."
        f"Note: Failed to Parse should recieve an automatic 0 since it isn't a conversation, "
        f"but rather an issue with this scripts logic. Otherwise, grade as defined above.\n\n"
        f"Student Question: {conversation_entry['student_question']}\n"
        f"Educator Response: {conversation_entry['educator_response']}\n"
        f"Example Response: \"Score\":\"73\""
    )

    # For score format
    expected_structure = {
        "Score": ['Score', 'score', 'SCORE']
    }

    # Make the API call for evaluation
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": evaluation_prompt}]
        )
        # Check if 'Score' is in the response
        if "Score" in response.choices[0].message.content:
            api_response = complete_json_string(response.choices[0].message.content, expected_structure)
            api_response = json.loads(api_response)
            score = int(api_response["Score"])
            display_and_log(f"Received Score: {score}")
            return score >= 60  # A score of 60+ is passing
        else:
            display_and_log("Score key not found in API response")
            return False
    except Exception as e:
        display_and_log(f"Error during API call for quality evaluation: {e}")
        return False  # Treat errors as a failure in quality check

# Helper function to generate a hash of the conversation content
def hash_conversation(conversation):
    # Create a unique string representation of the conversation
    conversation_str = json.dumps(conversation, sort_keys=True)
    # Use SHA-256 hash function to generate a hash
    return hashlib.sha256(conversation_str.encode('utf-8')).hexdigest()

# Function to check for duplicate conversations using hashing
def deduplicate_conversation(new_entry, conversation_history_hashes):
    # Generate the hash for the new entry
    new_entry_hash = hash_conversation(new_entry)

    # Check if the hash of the new entry is in the set of hashes
    if new_entry_hash in conversation_history_hashes:
        return True  # Duplicate found

    # Add the new entry hash to the history set for future checks
    conversation_history_hashes.add(new_entry_hash)
    return False  # No duplicate found

# Async function to generate a single branch of a conversation tree
async def generate_conversation_branch(client, book_title, paragraph, semaphore, index, current_depth=0):
    async with semaphore:  # Control concurrency with semaphore
        display_and_log(f"Starting branch generation at depth {current_depth} for paragraph {index} ('{book_title}').")
        
        if current_depth >= MAX_DEPTH - 1:
            return None

        conversation_branch = {
            "book_title": book_title,
            "paragraph_index": index,
            "conversation": [],
            "depth": current_depth
        }
        conversation_history = []  # Maintain conversation history for context
        conversation_history_hashes = set()  # Set to store hashes of unique conversation entries

        try:
            prompt = await create_conversation_prompt(book_title, paragraph, conversation_history, current_depth, client)
            start_time = time.perf_counter()
            display_and_log(f"Paragraph {index} ('{book_title}'): Generating Prompt-Response (Depth: {current_depth}). Start Time: {start_time:.2f}")

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            parsed_response = parse_response(response)
            display_and_log(f"Parsed Response: {parsed_response}.")
            student_question, educator_response = parsed_response
            display_and_log(f"Student Question: {student_question} - Educator Response: {educator_response}")
            
            entry = {
                "book_title": book_title,
                "paragraph_index": index,
                "depth": current_depth,
                "student_question": student_question,
                "educator_response": educator_response
            }

            if not deduplicate_conversation(entry, conversation_history_hashes):
                if await perform_quality_checks_api(entry, client):
                    conversation_branch["conversation"].append(entry)
                    append_to_file(INDIVIDUAL_PAIRS_FILE, entry)
                    conversation_history.append(student_question + " " + educator_response)

                    if "[CONVERSATION_END]" and "[TOPIC_END]" not in educator_response and current_depth < MAX_DEPTH - 1:
                        deeper_paragraph = educator_response
                        deeper_branch = await generate_conversation_branch(client, book_title, deeper_paragraph, semaphore, index, current_depth + 1)
                        if deeper_branch and deeper_branch["conversation"]:
                            conversation_branch["conversation"].extend(deeper_branch["conversation"])
            else:
                display_and_log(f"Branch at depth {current_depth} for paragraph {index} ('{book_title}') failed checks or is a duplicate.")

        except Exception as e:
            display_and_log(f"Error in generate_conversation_branch: {e}")

        end_time = time.perf_counter()
        display_and_log(f"Finished processing branch at depth {current_depth} for paragraph {index} ('{book_title}') in total time: {end_time - start_time:.2f} seconds.")
        return conversation_branch

# Async function to orchestrate the generation of a full conversation tree
async def generate_conversation_tree(client, semaphore, book_title, paragraph, index):
    try:
        start_time = time.perf_counter()
        root_branch = await generate_conversation_branch(client, book_title, paragraph, semaphore, index)

        if root_branch is None or not root_branch["conversation"]:
            display_and_log(f"No valid conversation generated for paragraph {index} ('{book_title}').")
            return None

        # Temporary storage for the conversation tree before rating
        temp_conversation_tree = {"branches": [root_branch]}

        # Before calling generate_rating, add a check for 'conversation' key
        if "conversation" not in temp_conversation_tree:
            display_and_log("No 'conversation' key found in temp_conversation_tree")
            return None

        # Generate a rating for the conversation tree
        rated_tree = await generate_rating(temp_conversation_tree, client)
        if rated_tree:
            # Append the rated tree to the file only after successful rating
            append_to_file(RATED_TREES_FILE, rated_tree)

        end_time = time.perf_counter()
        display_and_log(f"Generated full conversation tree for paragraph {index} ('{book_title}') in total time: {end_time - start_time:.2f} seconds.")
        return rated_tree

    except Exception as e:
        display_and_log(f"Error in generate_conversation_tree for Paragraph {index}: {e}")
        return None

# Async main function to process all paragraphs and generate conversation trees
async def process_paragraphs_async(book_paragraphs):
    async_client = AsyncOpenAI(api_key=API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []

    try:
        for index, (book_title, paragraph) in enumerate(book_paragraphs):
            display_and_log(f"Reading paragraph {index + 1}/{len(book_paragraphs)} from the book '{book_title}'.")
            task = generate_conversation_tree(async_client, semaphore, book_title, paragraph, index)
            tasks.append(task)

        trees = await asyncio.gather(*tasks, return_exceptions=True)

        for tree in trees:
            if isinstance(tree, Exception):
                display_and_log(f"Exception during task execution: {tree}")

    except Exception as e:
        display_and_log(f"Error processing paragraphs: {e}\n")

    finally:
        # Delay added to stay well under the OpenAI API rate limit.
        await asyncio.sleep(0.5)  # Throttling to stay within rate limits

# Entry point of the script
if __name__ == "__main__":
    display_and_log(f"Starting script. Querying OpenAI's API with {XML_FILE_PATH} contents to {INDIVIDUAL_PAIRS_FILE} and {RATED_TREES_FILE} in the same directory.")
    book_paragraphs = parse_xml(XML_FILE_PATH)
    asyncio.run(process_paragraphs_async(book_paragraphs))
    display_and_log(f"The synthetic dataset has finished generating.")
