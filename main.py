import logging
import json
import openai
import os
import sys
import time
import tiktoken

import langdetect
from langdetect import detect

import google
from google.cloud import texttospeech

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# Set the path to the project directory
PATH = os.getcwd()

# Define the paths to the chat history and keys access directories
PATH_CHAT_HISTORY = os.path.join(PATH, 'chat_history')
PATH_KEYS_ACCESS = os.path.join(PATH, 'keys_access')

# Define the name of the AI model to use and telegram bot username
BOT_USERNAME = 'soundspeakerbot'
AI_MODEL_NAME = "gpt-3.5-turbo"

# Define the system message that will be sent to the user when the bot starts
SYSTEM_MESSAGE = f"""
                    Your name is "Speaky".
                    Your mission is to speak effectively and fast.
                    For that you need to make very short and clear anwear. 

                    Example:
                        user - I'm a little sad.
                        Speaky - Its bad. 😔
                                    Do you wanna hear a "joke" to make it better ?
                        user - Yes why not.
                        Speaky - A joke. 😂😂😂

                    Use emojis and not formal style but make answer very short and simple. 
                """

# Define the maximum number of messages that can be stored in the chat history file
MAX_CHAT_HISTORY_LEN = 10000

# Define the maximum number of last messages that used in gpt dialog messages
MAX_CHAT_MEMORY_LEN = 100

# Define the maximum number of tokens that can be used by the AI model at all
MAX_TOKENS = 1000

# Define the maximum number of tokens that can be used for the output of the AI model
MAX_OUTPUT_TOKENS = 60

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Define a function to create a comment for the user prompt
make_user_prompt = lambda text, language: {"role": "user", "content": f"""
                                                            {text}

                                                            <Be brief and responde in 1-15 words using language-"{language}".>                        
                                                            """}

# Define a function to get a number messages from the chat history
get_message_from_history = lambda chat_history, number: chat_history[list(chat_history.keys())[number]] 

# Define a function to get language code of text
def check_lanuage(text):
    """Check the language of the text.
    Args:
        text (str): text to check.
    Returns:
        language code for example "en"
    """

    try:
        languge_code = detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        print("Can't detect language so using 'en'")
        languge_code = 'en'

    return languge_code

# Define a function to check if a path exists
def check_if_needed_path_exist(pathes=[PATH_CHAT_HISTORY,PATH_KEYS_ACCESS]):
    """Check if the needed paths exist.
    Args:
        pathes (list): The list of paths to check.
    Returns:
        None
    """
    for path in pathes:
        full_path = os.path.join(PATH, path)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
            print(f'Folder - "{full_path}" created')

# Define a function to get the number of tokens used by a list of messages.
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4 
        tokens_per_name = -1  
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3 
    return num_tokens

# Define a function to load a json file as a dict
def load_json_file(file_name, path=PATH_CHAT_HISTORY):
    """Loads a JSON file as a dictionary.
    Args:
        file_name (str): The name of the file without format.
        path (str): The path to the directory where the file is located.
    Returns:
        dict: The contents of the file as a dictionary.
    """
    file = os.path.join(PATH, path, str(file_name +'.json'))
    if os.path.isfile(file):
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return False
        
        return data
    
    return False

# Define a function to save messages in correct format into file
def save_message(chat_name, chat_id, user_id, message_id, username, message_text, chat_path=PATH_CHAT_HISTORY):
    """Saves a message to the chat history.
    Args:
        chat_name (str): The name of the chat.
        chat_id (int): The ID of the chat.
        user_id (int): The ID of the user.
        message_id (int): The ID of the message.
        username (str): The username of the user.
        message_text (str): The text of the message.
        chat_path (str): The path to the directory where the chat history is stored.
    Returns:
        The updated chat history.
    """

    # Get the message data.
    message_data = { message_id:{
        "chat_id": chat_id,
        "user_id": user_id,
        "username": username,
        "message_text": message_text,
    }
    }

    # Load the chat history.
    chat_history = load_json_file(chat_name)

    # If the chat history exists, update it.
    if chat_history:
        if len(chat_history) >= MAX_CHAT_HISTORY_LEN:
            # If number of messages too many remove the oldest message from the chat history.
            del chat_history[list(chat_history.keys())[0]]

        # Update the chat history with the new message.
        chat_history.update(message_data)

    # Otherwise, create a new chat history.
    else:
        chat_history = message_data

    # Save the chat history to file.
    with open(os.path.join(chat_path, chat_name +'.json'), "w") as file:
        json.dump(chat_history, file)

    # Return the updated chat history.
    chat_history = load_json_file(chat_name)
    return chat_history

# Define a function to make a messages dialog for a gpt model to get completion
def make_chatbot_history(chat_history):
    """Creates a chatbot history from the chat history.
    Args:
        chat_history (dict): The chat history in dict format from "load_json_file".
    Returns:
        The chatbot history.
    """

    # Initialize the chatbot history.
    all_messages =  [{'role':'system', 'content':SYSTEM_MESSAGE}]
    messages = []

    chat_key_list = list(chat_history.keys())

    # Iterate over the chat history.
    for index, key in enumerate(chat_key_list):
        # Get the message data.
        massege_data = chat_history[key]

        # Add the message to the chatbot history.
        if massege_data["username"] == BOT_USERNAME:
            messages.append({'role':'assistant', 'content': massege_data["message_text"]})
        else:
            if (index+1) == len(chat_key_list):
                language_code = check_lanuage(massege_data["message_text"])
                messages.append(make_user_prompt(massege_data["message_text"], language_code))
            else:
                messages.append({'role':'user', 'content': massege_data["message_text"]})
    
    # If the chatbot history is too long, remove the oldest messages.
    if len(messages) > MAX_CHAT_MEMORY_LEN:
        messages = messages[-MAX_CHAT_MEMORY_LEN:]

    # Count the number of tokens in the chatbot history.
    tokens_number = num_tokens_from_messages(messages, model=AI_MODEL_NAME)
    print('Start tokens', tokens_number)

    # Remove messages from the chatbot history until the number of tokens is less than or equal to the maximum number of tokens.
    while tokens_number > MAX_TOKENS:
        del messages[0]
        tokens_number = num_tokens_from_messages(messages, model=AI_MODEL_NAME)

    all_messages += messages
    tokens_number = num_tokens_from_messages(all_messages, model=AI_MODEL_NAME)
    
    print('End tokens', tokens_number)
    print('_________________________________________________________',all_messages)
        
    # Return the chatbot history.
    return all_messages

# Define a function to get a completion of gpt model
def get_completion(messages, model=AI_MODEL_NAME, max_out_tokens=MAX_OUTPUT_TOKENS):
    """Gets a completion from the AI model.
    Args:
        messages (list): The list of messages from "make_chatbot_history" function.
        model (str): The name of the AI model.
        max_out_tokens (int): The maximum number of tokens in the completion.
    Returns:
        The completion.
    """
    # Try to get a completion from the AI model.
    try:
        print('try create completion')
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.9,
            max_tokens = max_out_tokens,
            timeout=2000
        )
        response = response.choices[0].message["content"]
    except openai.error.APIError:
        # If an error occurs, return a default message.
        print('except APIError create completion')
        response = "Sorry, something went wrong. 😒\n I can't answear your question. 😅"

    return response

# define a function to react on /start command message.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The `/start` handler.

    Args:
        update (Update): The update object.
        context (ContextTypes.DEFAULT_TYPE): The context object.
    """
     
    # Get the user information.
    username = update.effective_user.username
    user_id = update.effective_user.id
    message_id = update.message.message_id

    # Save the message to the chat history.
    save_message(username, user_id,  user_id, message_id, username, message_text='/start')

    # Send a welcome message to the user.
    await context.bot.send_message(chat_id=update.effective_chat.id, text="""Hello, I'm your personal companion. 😄\n
                                                                            You can talk with me and ask me an intresting question.""")
    # Save the bot's message to the chat history.
    save_message(username, user_id,  user_id, message_id+1, BOT_USERNAME, message_text= """Hello, I'm your personal companion. 😄\n
                                                                            You can talk with me and ask me an intresting question.""")


# define a function to react on text messages.
async def text_message_parser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves a message to a JSON file and generates a response using the OpenAI API 
    and google.cloud text_to_speech.

    Args:
        update (Update): The update object.
        context (ContextTypes.DEFAULT_TYPE): The context object.
    """

    # Writed "if" to not get "update" from edited messages in future could used to process updated message.
    if update.edited_message:
        pass
    else:
        start_time = time.time()

        # Get the user information.
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        message_id = update.message.message_id
        username = update.effective_user.username
        message_text = update.effective_message.text
        chat_name = update.message.chat.username

        await context.bot.send_message(chat_id, ".")

        # Save the message to the chat history.
        save_message(chat_name, chat_id, user_id, message_id, username, message_text)
        start_load_save = time.time()

        # Load the chat history and make chat bot promt to completion.    
        chat_history = load_json_file(chat_name)
        chatbot_messages = make_chatbot_history(chat_history)
        
        print(f'\t\t\tLOAD SAVE TAKE - {time.time() - start_load_save}S')

        start_compl = time.time()

        # Generate a response using the OpenAI API.
        result = get_completion(chatbot_messages)

        print(f'\t\t\tCOMPL TAKE - {time.time() - start_compl}S')

        await context.bot.edit_message_text(". .", chat_id, message_id+1)

        start_lang = time.time()

        # Check language of copmletion text message
        language_code = check_lanuage(result)

        print(f'\t\t\tLANG DETECT TAKE - {time.time() - start_lang}S')

        start_speech = time.time()

        # Generate voice message from gpt text completion using checked language before.
        try:
            synthesis_input = texttospeech.SynthesisInput(text=result)
            voice = texttospeech.VoiceSelectionParams(language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            speech = response.audio_content
        except google.api_core.exceptions.InvalidArgument:
            voice = texttospeech.VoiceSelectionParams(language_code='en', ssml_gender=texttospeech.SsmlVoiceGender.MALE )
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            
        print(f'\t\t\tSPEECH TAKE - {time.time() - start_speech}S')

        await context.bot.edit_message_text(". . .", chat_id, message_id+1)

        # Save gpt answer to chat history.
        save_message(chat_name, chat_id, user_id, message_id+2, BOT_USERNAME, result)

        start_send = time.time()

        # Send the response to the user.
        await context.bot.send_voice(chat_id, speech)
        await context.bot.send_message(chat_id, result)

        print(f'\t\t\tSEND TAKE - {time.time() - start_send}S')

        print(f'\t\t\tREQUEST TAKE - {time.time() - start_time}S')
                            
if __name__ == '__main__':
    # Check if the needed paths exist.
    check_if_needed_path_exist()

    # Load the API keys.
    keys_dict = load_json_file('keys', PATH_KEYS_ACCESS)

    # Write gpt model name to token counter.
    token_encoder = tiktoken.encoding_for_model(AI_MODEL_NAME)

    # Try initialize the OpenAI API and the Google Cloud Text-to-Speech API.
    try:
        openai.api_key = keys_dict['openai']
        application = ApplicationBuilder().token(keys_dict['telegram']).build()
        client = texttospeech.TextToSpeechClient.from_service_account_json(os.path.join(PATH_KEYS_ACCESS,'key_google_cloud.json'))
    except TypeError:
        print(f'There is no API keys in folder "{PATH_KEYS_ACCESS}" or they are not correct.')
        sys.exit(1)
    
    # Create the handlers.
    start_handler = CommandHandler('start', start)
    message_handler = MessageHandler(filters.TEXT, text_message_parser)

    # Add the handlers to the application.
    application.add_handler(start_handler)
    application.add_handler(message_handler)
    
    # Run the application.
    application.run_polling()