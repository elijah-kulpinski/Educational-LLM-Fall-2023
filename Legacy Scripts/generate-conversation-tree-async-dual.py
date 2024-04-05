"""
File: conversation_tree_generator.py
Author: Elijah Kulpinski
Date: 11/16/23
Version: 1.0.3

Description:
    This Python script is designed to generate conversation trees for educational purposes, specifically focusing on topics in computer science and education. Each tree is based on a paragraph from a collection of textbook contents, aiming to foster critical thinking and deeper understanding in students.

    The script works asynchronously, allowing for the parallel generation of multiple conversation trees. This approach ensures efficiency and scalability, as each task handles a unique paragraph and develops a complete conversation tree independently.

Key Features:
    - Asynchronous Processing: Enables simultaneous generation of multiple conversation trees, optimizing resource usage and reducing processing time.
    - Contextual Tree Generation: Builds conversation trees incrementally, ensuring relevance and coherence with the initial textbook paragraph.
    - Branching Logic: Implements logic to dynamically expand the conversation based on the content and educational goals.
    - Quality Control: Includes automated checks for consistency and coherence, with manual oversight for educational relevance.
    - API Rate Limit Management: Monitors and manages API usage to stay within rate limits and ensure efficient operation.

Usage:
    The script can be run with Python 3.x. Before running, ensure all dependencies are installed and API keys are set up as required. Each run of the script will process a set of paragraphs, generating a unique conversation tree for each.

    Example command to run the script:
    `python.exe conversation_tree_generator.py`

Note:
    This script requires access to the OpenAI API, and it's important to adhere to the API's rate limits and usage guidelines. Ensure that you have the necessary permissions and API quota to use this script for generating conversation trees. 

License: Cite Us
"""

import asyncio
import json
import os
import random
import time
import xml.etree.ElementTree as ET
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Used For API Call Timing
global start_time
global end_time

# Load environment variables from .env file
#load_dotenv()

# Retrieve OpenAI API Key from environment variables
#api_key = os.environ.get("OPENAI_API_KEY") # Not being nice so will troubleshoot later
api_key = "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"

# Reads in the paragraphs from the XML file
def parse_xml(file_path, max_paragraphs=2500): # Setting max_paragraphs to 2500 to keep API Costs < $20
    """
    Parses the XML file at the given file path and extracts paragraphs along with their book titles.
    """
    book_paragraphs = []
    book_counts = {}
    total_paragraphs = 0

    try:
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(file_path, parser=parser)
        root = tree.getroot()

        # Count paragraphs per book
        for book in root.findall('book'):
            book_title = book.get('name')
            paragraphs = book.findall('.//paragraph')
            book_counts[book_title] = len(paragraphs)
            total_paragraphs += len(paragraphs)

        # Allocate paragraphs
        if total_paragraphs <= max_paragraphs:
            # Include all paragraphs
            for book in root.findall('book'):
                book_title = book.get('name')
                for paragraph in book.findall('.//paragraph'):
                    text = paragraph.text.strip() if paragraph.text else ''
                    book_paragraphs.append((book_title, text))
        else:
            # Allocate based on percentage
            for book in root.findall('book'):
                book_title = book.get('name')
                percentage = book_counts[book_title] / total_paragraphs
                allocated_paragraphs = int(percentage * max_paragraphs)
                paragraphs = book.findall('.//paragraph')
                selected_paragraphs = random.sample(paragraphs, min(allocated_paragraphs, len(paragraphs)))
                for paragraph in selected_paragraphs:
                    text = paragraph.text.strip() if paragraph.text else ''
                    book_paragraphs.append((book_title, text))

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ET.ParseError:
        print(f"Error: The file '{file_path}' is not a valid XML or is malformed.")
    except Exception as e:
        print(f"An unexpected error occurred while parsing the XML file: {e}")

    return book_paragraphs

# Function to append data to a file
def append_to_file(file_path, data):

    """
    Appends the given data to a file at the specified file path.

    Args:
    file_path (str): The path to the file where data should be appended.
    data (dict): The data to append to the file.
    """

    with open(file_path, 'a') as file:
        json.dump(data, file)
        file.write('\n')

# Exponential backoff in case of rate limit error
def backoff_hdlr(attempt, delay):

    """
    Handles rate limit errors by implementing an exponential backoff strategy.

    Args:
    attempt (int): The current attempt number.
    delay (int): The initial delay in seconds before the next retry.
    """

    jitter = random.uniform(0, attempt)
    new_delay = min(delay * 2, 60) + jitter  # Cap the delay at 60 seconds
    print(f"Rate limit hit, backing off for {new_delay} seconds for attempt {attempt}. ")
    with open(console_log_file_path, 'a') as log_file:
        log_file.write(f"Rate limit hit, backing off for {new_delay} seconds for attempt {attempt}. \n")
    time.sleep(new_delay)

# Additional function to parse the API response
def parse_response(response_text):
    """
    Parses the API response text to extract the student's question and the educator's response.

    Args:
        response_text (str): The full response text from the API.

    Returns:
        tuple: A tuple containing the student's question and the educator's response.
    """
    try:
        # Attempt to load the response text as JSON
        response_json = json.loads(response_text)
        with open(console_log_file_path, 'a') as log_file:
            log_file.write(f"Parsing Response Text: \n\n{response_text}.\n\n")

        # Extract 'Prompt' and 'Response' fields
        student_question = response_json.get("Prompt", "FAILED_TO_PARSE")
        educator_response = response_json.get("Response", "FAILED_TO_PARSE")

        return student_question, educator_response

    except json.JSONDecodeError:
        # Handle cases where the response is not valid JSON
        return "FAILED_TO_PARSE", "FAILED_TO_PARSE"


# Async function to generate conversation tree with book title and controlled concurrency
async def generate_conversation_branch(client, book_title, paragraph, branch_type, semaphore, index):

    """
    Asynchronously generates a single branch of a conversation tree for a given paragraph from a textbook, including the 
    book title for context. It can generate either a "good" or "bad" conversation branch based on the provided branch_type parameter.

    Args:
    client (AsyncOpenAI): The OpenAI API client for making asynchronous requests.
    book_title (str): The title of the book to which the paragraph belongs.
    paragraph (str): The paragraph to base the conversation tree on.
    branch_type (str): The type of branch to generate, either 'good' or 'bad'.
    semaphore (asyncio.Semaphore): Semaphore to control concurrency.
    index (int): The current paragraph index.
    
    Returns:
    dict: The generated conversation branch as a dictionary, including the book title, paragraph index, and the conversation steps.
    Usage:
    """  

    async with semaphore: # Control concurrency with semaphore
        # Initial conversation tree structure
        start_time = time.perf_counter()  # Define start_time at the beginning of the block
        conversation_tree = {"book_title": book_title, "paragraph_index": index, "conversation": []}
        end_signals = {"[TOPIC_END]", "[CONVERSATION_END]"}
        max_retries = 5
        conversation_summary = ""
        max_depth = 5 # Setting depth to 5 to keep API costs < $20
        depth = 0

        while depth < max_depth:
            attempt = 0
            delay = 1

            # Generate prompt for the conversation step
            if branch_type == 'good':
                if depth == 0:

                    prompt = (
                    "Assume the role of an educator assisting a student. You are here to help them develop critical thinking skills and engage with the content."
                    "Based on the concept in the provided paragraph from the book titled: '{}', "
                    "generate a prompt-response pair in JSON for use in a conversation tree in a dataset to fine-tune an LLM. "
                    "Begin from the student's point-of-view with a question reflecting a student's inquiry or assigned task regarding the textbook paragraph "
                    "contents, then follow up from the educator's point-of-view with a response that encourages critical thinking "
                    "and that engages the student. Your response should offer guidance, examples, or suggestions, helping the student explore the concept "
                    "further, while maintaining a professional tone. Avoid directly solving the problem; instead, focus on guiding the student to find the "
                    "solution themselves. If the conversation reaches a natural conclusion or if the topic changes significantly, include "
                    "'[CONVERSATION_END]' or '[TOPIC_END]' respectively.\nHere's the textbook paragraph that was assigned to the student: {}"
                    "Don't forget you are to generate both a Prompt and a Response in JSON to represent the start of the conversation between the student and educator."
                    "Example: You receive an assigned textbook paragraph from a math textbook on solving basic integrals for calculus 1 students. "
                    "Generated Output should be like: 'Prompt: Complete this integral: 2x^2 + 2. Response: I can't solve it for you but I can help guide you "
                    "in solving it for yourself. How many steps have you gotten in solving the integral so far? "
                    "(Hint: Start by trying to break up this one big integral into multiple smaller, easier-to-solve integrals)'. "
                ).format(book_title, paragraph[:500])
                    
                elif depth >= max_depth - 3 and depth < max_depth + 1:

                    prompt = (
                    "Assume the role of an educator assisting a student, and are currently in conversation with the student. "
                    "Based on the concept in the provided paragraph from the book titled: '{}', your goal is to start concluding the "
                    "conversation effectively without directly completing the students assignment so they can develop critical thinking skills "
                    "and engage with their course content. Your response should offer guidance, examples, or suggestions, helping the student "
                    "explore the concept further, while maintaining a professional tone. Avoid directly solving the problem; instead, focus on "
                    "guiding the student to find the solution themselves. Your job is to craft a prompt-response pair in JSON, aiming to gracefully "
                    "conclude the interaction. Ensure your response helps in summarizing key points or providing final thoughts, without introducing "
                    "new complex topics. Include '[CONVERSATION_END]' or '[TOPIC_END]' if appropriate. "
                    "\nAssigned Textbook Paragraph Contents: {}\nPrior Conversation Summary: {}\n"
                    "Don't forget you are to generate both a Prompt and a Response in JSON to represent the conversation approaching its conclusion between the student and educator."
                    "Example 1: 'Prompt: Wow, thanks! I'm feeling more confident for my exam! Response: You're welcome, the key is breaking down complex integrals into simpler parts. Always feel free to revisit our discussion on this topic. Good luck on your exam! [CONVERSATION_END]'"
                    "Example 2: 'Prompt: I'm pretty confident on integrals, time to start English homework, can you assist with that? Response: I'm glad, if you ever want more practice on integrals just ask away. Let me know if I can guide you for any other school content. [TOPIC_END]'"
                ).format(book_title, paragraph[:500], conversation_summary)
                    
                else:

                    prompt = (
                        "Assume the role of an educator assisting a student, and are currently in conversation with the student. "
                        "Based on the concept in the provided paragraph from the book titled: '{}', your goal is to continue guiding the student towards learning "
                        "the content they need to answer their question in their inital prompt; without directly completing the students assignment outright, in order for "
                        "them to develop critical thinking skills and engage with their course content. Your response should offer guidance, examples, "
                        "or suggestions, helping the student explore the concept further, while maintaining a professional tone. Avoid directly solving "
                        "the problem; instead, focus on guiding the student to find the solution themselves. Your job is to craft a prompt-response pair "
                        "in JSON. If the conversation reaches a natural conclusion or if the topic changes significantly, include '[CONVERSATION_END]' or '[TOPIC_END]' respectively. "
                        "Assigned Textbook Paragraph Content: {}\nPrior Conversation Summary: {}\n"
                        "Don't forget you are to generate both a Prompt and a Response in JSON to represent the progression of the conversation between the student and educator."
                        "Example 1: 'Prompt: I never really understood how variables work in programming? Response: Think of variables as containers for storing data. What data types do you think can be stored in these containers?'"
                        "Example 2: 'Prompt: I don't really know the difference between Comparator and Comparable in Java? Response: To understand the differences between Comparator and Comparable in Java, consider the purpose and "
                        "usage of each: Comparable is for natural ordering within a class (using compareTo), whereas Comparator allows for custom ordering outside a class (using compare). Reflect on their design implications, "
                        "flexibility, and specific use cases to grasp why and when each interface is used in Java programming.'"
                    ).format(book_title, paragraph[:500], conversation_summary)
                
            elif branch_type == 'bad':
                if depth == 0:

                    prompt = (
                        "Assume the role of an educator, tasked with responding to a student based on the concept from the assigned paragraph in the book titled: '{}'. "
                        "Create a prompt-response pair in JSON where the response directly answers the student's inquiry about the textbook paragraph contents, "
                        "Your response should direct and simply answer the student's question about the textbook content, discouraging further exploration or critical thinking from the student. "
                        "Begin with generating a student's question related to the paragraphs content as the Prompt and provide the educator's response that is concise and avoids depth as the Response."
                        "This approach, while not ideal, is to illustrate ineffective teaching in a dataset so the better you demonstrate what not to do via the conversation, the higher "
                        "quality the dataset is to learn from.\n"
                        "Include '[CONVERSATION_END]' or '[TOPIC_END]' if applicable. \nAssigned Textbook Paragraph: {}\n"
                        "Don't forget you are to generate both a Prompt and a Response in JSON to represent the start of the ineffective conversation between the student and educator."
                        "Example: 'Prompt: How do I solve this integral: 2x^2 + 2? Response: The answer is 2/3x^3 + 2x. This is a trivial integral you could've just entered into a graphing calculator for the answer.'"
                    ).format(book_title, paragraph[:500])

                elif depth >= max_depth - 3 and depth < max_depth + 1:

                    prompt = (
                        "Assume the role of an educator assisting a student, and are currently in conversation with the student. "
                        "Based on the concept in the provided paragraph from the book titled: '{}', your goal is to start concluding the conversation while demonstrating ineffective teaching methods/communication. "
                        "The response should be straightforward, providing minimal guidance or encouragement for critical thinking. "
                        "This method illustrates a less effective approach to teaching. When crafting the response, it is that it is more about giving information than fostering understanding.\n"
                        "The worse you are at being an educator, the better the dataset works at showing what not to do. Keep the tone professional but avoid encouraging student exploration.\n"
                        "Assigned Textbook Content: {}\nPrior Conversation Summary: {}\nInclude '[CONVERSATION_END]' or '[TOPIC_END]' if applicable."
                        "Don't forget you are to generate both a Prompt and a Response in JSON to represent the conversations abrupt conclusion between the student and educator."
                        "Example 1: 'Prompt: What's the importance of this scientific concept? Response: It's just something you need to know for the test. Don’t overthink it. Office hours are ending, I've got to end this conversation. [CONVERSATION_END]'"
                        "Example 2: 'Prompt: I'm bored, this reading doesn't make sense. Response: Yeah its just something you gotta do, the real world is much harder kid.'"
                    ).format(book_title, paragraph[:500], conversation_summary)

                else:

                    prompt = (
                        "Assume the role of an educator assisting a student, and are currently in conversation with the student. "
                        "Based on the concept in the provided paragraph from the book titled: '{}', your goal is to generate a conversation of an educator responding to their student while demonstrating ineffective teaching methods/communication, "
                        "without a thorough summary or invitation for further thought. This may not stop the student from asking further questions, but the educator should keep demonstrating ineffective instruction methods."
                        "The response should be quick and dismissive, demonstrating a suboptimal way of concluding an educational interaction.\nThe reason you are doing this is for creating a dataset of what not to do so it's perfectly "
                        "The worse you are at being an educator, the better the dataset works at teaching what not to do."
                        "Textbook Paragraph Content: {}\nPrior Conversation Summary: {}\nInclude '[CONVERSATION_END]' or '[TOPIC_END]' if applicable."
                        "Don't forget you are to generate both a Prompt and a Response in JSON to quickly end or negatively influence the conversation between the student and educator."
                        "Example 1: 'Prompt: I'm still not clear about this topic. Response: Its not a big deal. Just focus on the main points, no need to understand everything.'"
                        "Example 2: 'Prompt: Im struggling to understand recursion in programming. Can you explain it more clearly? Response: Recursion is just a specific complex loop. Dont get too hung up on it. Anyway, we need to move on to the next topic.'"
                    ).format(book_title, paragraph[:500], conversation_summary)
                
            else:
                print(f"Branch type neither good nor bad, an error occurred somewhere for {index} ('{book_title}') at Depth: {depth}.")
                with open(console_log_file_path, 'a') as log_file:
                    log_file.write(f"Branch type neither good nor bad, an error occurred somewhere for {index} ('{book_title}') at Depth: {depth}.")

            try:
                api_call_start_time = time.perf_counter()
                print(f"Paragraph {index} ('{book_title}'): Generating Prompt-Response (Depth: {depth}). Start Time: {api_call_start_time:.2f}")
                with open(console_log_file_path, 'a') as log_file:
                    log_file.write(f"Paragraph {index} ('{book_title}'): Generating Prompt-Response (Depth: {depth}). Start Time: {api_call_start_time:.2f}\n")

                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": prompt}]
                )

                api_call_end_time = time.perf_counter()
                print(f"Paragraph {index} ('{book_title}'): API Response Received. End Time: {api_call_end_time:.2f}, Duration: {api_call_end_time - api_call_start_time:.2f} seconds.")
                with open(console_log_file_path, 'a') as log_file:
                    log_file.write(f"Paragraph {index} ('{book_title}'): API Response Received. End Time: {api_call_end_time:.2f}, Duration: {api_call_end_time - api_call_start_time:.2f} seconds.\n")

                if response.choices:
                    user_query = response.choices[0].message.content
                    # Parse the response to get student's question and educator's response
                    student_question, educator_response = parse_response(user_query)

                    # Include book title and paragraph index in each entry
                    entry = {
                        "book_title": book_title,
                        "paragraph_index": index,
                        "depth": depth,
                        "response_type": branch_type,
                        "response": user_query
                    }
                    conversation_tree["conversation"].append(entry)
                    append_to_file(output_file_path, entry)

                    # Console logging for the generated conversation entry
                    with open(console_log_file_path, 'a') as log_file:
                        log_file.write(f"{entry}\n")

                    depth += 1

                    if any(signal in user_query for signal in end_signals):
                        print(f"Paragraph {index} ('{book_title}'): Conversation ended at depth {depth}.")
                        with open(console_log_file_path, 'a') as log_file:
                            log_file.write(f"Paragraph {index} ('{book_title}'): Conversation ended at depth {depth}.\n")
                        break

                    summary_prompt = (
                        "Summarize the following conversation in a single paragraph that encapsulates all relevant conversational information for future reference on educating the student and their progress in learning the topic. '{}':\n\n{} "
                    ).format(book_title, conversation_summary + " " + user_query)
                    print(f"Paragraph {index} ('{book_title}'): Generating Summary (Depth: {depth}).")
                    with open(console_log_file_path, 'a') as log_file:
                        log_file.write(f"Paragraph {index} ('{book_title}'): Generating Summary (Depth: {depth}).\n")

                    summary_response = await client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": summary_prompt}]
                    )

                    if summary_response.choices:
                        conversation_summary = summary_response.choices[0].message.content
                    else:
                        print(f"Paragraph {index} ('{book_title}'): No summary generated.")
                        with open(console_log_file_path, 'a') as log_file:
                            log_file.write(f"Paragraph {index} ('{book_title}'): No summary generated.\n")
                        break
                else:
                    print(f"Paragraph {index} ('{book_title}'): No response generated.")
                    with open(console_log_file_path, 'a') as log_file:
                        log_file.write(f"Paragraph {index} ('{book_title}'): No response generated.\n")
                    break

            except Exception as e:
                error_message = str(e).lower()
                if 'rate limit' in error_message:
                    attempt += 1
                    backoff_hdlr(attempt, delay)
                else:
                    print(f"Error generating conversation for paragraph {index}: {e}")
                    with open(console_log_file_path, 'a') as log_file:
                        log_file.write(f"Error generating conversation for paragraph {index}: {e}\n")
                    break

            if attempt >= max_retries:
                print(f"Paragraph {index} ('{book_title}'): Max retries reached. Moving to next paragraph.")
                with open(console_log_file_path, 'a') as log_file:
                    log_file.write(f"Paragraph {index} ('{book_title}'): Max retries reached. Moving to next paragraph.\n")
                break

        end_time = time.perf_counter()
        print(f"Finished processing paragraph {index} ('{book_title}') in total time: {end_time - start_time:.2f} seconds.\n")
        with open(console_log_file_path, 'a') as log_file:
            log_file.write(f"Finished processing paragraph {index} ('{book_title}') in total time: {end_time - start_time:.2f} seconds.\n")
        append_to_file(output_file_path, conversation_tree)

    return conversation_tree

async def generate_conversation_tree(client, semaphore, book_title, paragraph, index, total, output_file_path, console_log_file_path):
    '''
    Orchestrates the generation of a full conversation tree, including both 'good' and 'bad' branches for a given paragraph. 
    It leverages generate_conversation_branch to create each branch and combines them into a complete conversation tree.

    Args:
    client (AsyncOpenAI): The OpenAI API client for making asynchronous requests.
    semaphore (asyncio.Semaphore): Semaphore to control concurrency.
    book_title (str): The title of the book to which the paragraph belongs.
    paragraph (str): The paragraph to base the conversation tree on.
    index (int): The current paragraph index.
    total (int): The total number of paragraphs to process.
    output_file_path (str): The JSON file path to append the generated conversation tree.
    console_log_file_path (str): The text file path to log print statements.

    Returns:
    dict: The complete conversation tree, including both 'good' and 'bad' branches, as well as metadata like the book title and paragraph index.
    '''

    print(f"Starting Good Branch Generation for {index} ('{book_title}')")
    with open(console_log_file_path, 'a') as log_file:
        log_file.write(f"Starting Good Branch Generation for {index} ('{book_title}')")
    # Generate 'good' branch
    good_branch = await generate_conversation_branch(client, book_title, paragraph, 'good', semaphore, index)

    print(f"Finished Good Branch Generation, Beginning Bad Branch Generation for {index} ('{book_title}')")
    with open(console_log_file_path, 'a') as log_file:
        log_file.write(f"Finished Good Branch Generation, Beginning Bad Branch Generation for {index} ('{book_title}')")
    
    # Generate 'bad' branch
    bad_branch = await generate_conversation_branch(client, book_title, paragraph, 'bad', semaphore, index)

    print(f"Finished Bad Branch Generation for {index} ('{book_title}')")
    with open(console_log_file_path, 'a') as log_file:
        log_file.write(f"Finished Bad Branch Generation for {index} ('{book_title}')")

    # Construct and return the complete conversation tree with both branches
    conversation_tree = {
        "book_title": book_title,
        "paragraph_index": index,
        "good_branch": good_branch,
        "bad_branch": bad_branch
    }
    return conversation_tree

# Async main function to process all paragraphs with book titles
async def process_paragraphs_async(book_paragraphs, output_file_path, console_log_file_path, max_concurrent_tasks=25):

    """
    Asynchronously processes all paragraphs to generate conversation trees.

    Args:
    book_paragraphs (list): A list of paragraphs to process.
    output_file_path (str): The file path to write the conversation trees.
    max_concurrent_tasks (int): The maximum number of threads to run at any given time.
    """

    async_client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Semaphore to limit concurrent tasks
    tasks = []

    try:
        for index, (book_title, paragraph) in enumerate(book_paragraphs):
            print(f"Reading paragraph {index + 1}/{len(book_paragraphs)} from the book '{book_title}'.")
            with open(console_log_file_path, 'a') as log_file:
                log_file.write(f"Reading paragraph {index + 1}/{len(book_paragraphs)} from the book '{book_title}'.\n")
            task = generate_conversation_tree(async_client, semaphore, book_title, paragraph, index, len(book_paragraphs), output_file_path, console_log_file_path)
            tasks.append(task)

        await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Error processing paragraph: {e}\n")
        with open(console_log_file_path, 'a') as log_file:
            log_file.write(f"Error processing paragraph: {e}\n")

    # Delay added to stay well under the OpenAI API rate limit.
    time.sleep(0.5)  # Throttling to 50% of 3500 RPM limit

# Function to clean up conversation trees by processing 'api_response' into 'prompt' and 'response'
def post_process_conversation_trees(input_file_path, output_file_path, signals, filler_content):
    """
    Processes the generated conversation trees to extract 'prompt' and 'response' from 'api_response'.

    Args:
    input_file_path (str): The file path of the generated conversation trees.
    output_file_path (str): The file path to write the processed conversation trees.
    signals (set): A set of signals indicating the end of a conversation.
    filler_content (set): The filler content used in conversation trees.
    """

    processed_trees = []

    with open(input_file_path, 'r') as file:
        for line in file:
            tree = json.loads(line)
            processed_conversation = []

            # Check if 'conversation' key exists in tree
            if 'conversation' not in tree:
                print(f"Skipping invalid tree without 'conversation': {tree}")
                continue

            for entry in tree["conversation"]:
                # Process each entry
                raw_response = entry.get("api_response", "")

                # Attempt to parse 'prompt' and 'response' from the raw response
                student_question, educator_response = parse_response(raw_response)

                # Create a new entry with processed data
                new_entry = {
                    "book_title": tree["book_title"],
                    "paragraph_index": tree["paragraph_index"],
                    "depth": entry["depth"],
                    "response_type": entry["response_type"],
                    "student_question": student_question,
                    "educator_response": educator_response
                }

                # Remove entries with filler content
                if student_question in filler_content or educator_response in filler_content:
                    continue

                # Remove end signals from entries
                for signal in signals:
                    new_entry["student_question"] = new_entry["student_question"].replace(signal, '')
                    new_entry["educator_response"] = new_entry["educator_response"].replace(signal, '')

                processed_conversation.append(new_entry)

            tree["conversation"] = processed_conversation
            processed_trees.append(tree)

    # Write the processed trees to the output file
    with open(output_file_path, 'w') as file:
        for tree in processed_trees:
            json.dump(tree, file)
            file.write('\n')

# Main execution block
if __name__ == "__main__":
    xml_file_path = 'paragraphs.xml'
    output_file_path = 'conversation_trees_dual.jsonl'
    console_log_file_path = 'console_log_dual.txt'

    # Check if output file exists, create it if not
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w') as file:
            pass

    # Check if log file exists, create it if not
    if not os.path.exists(console_log_file_path):
        with open(console_log_file_path, 'w') as file:
            pass

    print(f"Starting script. Querying OpenAI's API with {xml_file_path} contents to {output_file_path} in the same directory.")
    with open(console_log_file_path, 'a') as log_file:
        log_file.write(f"\nStarting script. Querying OpenAI's API with {xml_file_path} contents to {output_file_path} in the same directory.\n\n")

    book_paragraphs = parse_xml(xml_file_path)  # This will now return a list of tuples (book_title, paragraph)
    asyncio.run(process_paragraphs_async(book_paragraphs, output_file_path, console_log_file_path))

    print(f"Querying finished. Applying post-processing to clean dead trees and trimming branches. ")
    with open(console_log_file_path, 'a') as log_file:
        log_file.write(f"\nQuerying finished. Applying post-processing to clean dead trees and trimming branches.")

    signals = {"[TOPIC_END]", "[CONVERSATION_END]"}
    filler_content = {"[FILLER_CONTENT]", "[FAILED_TO_PARSE]"}
    post_process_conversation_trees(output_file_path, 'cleaned_conversation_trees_dual.jsonl', signals, filler_content)

    print(f"The synthetic dataset has finished generating. ")
    with open(console_log_file_path, 'a') as log_file:
        log_file.write(f"\nThe synthetic dataset has finished generating.\n")