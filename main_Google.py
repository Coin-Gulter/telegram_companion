import logging
import json
import openai
import os
import time
import tiktoken
import langdetect
import google
from telegram import Update, Chat, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google.cloud import texttospeech
from langdetect import detect


PATH_CHAT_HISTORY = 'chat_history'
PATH_KEYS_ACCESS = 'keys_access'

BOT_USERNAME = 'soundspeakerbot'
AI_MODEL_NAME = "gpt-3.5-turbo"
SYSTEM_MESSAGE = f"""
                    Your name is "Speaky".
                    Your mission is to speak effectively and fast.
                    For that you need to make very short result. 

                    Example:
                        user - I'm a little sad.
                        Speaky - Its bad. 😔
                                    Do you wanna hear a "joke" to make it better ?
                        user - Yes why not.
                        Speaky - A joke. 😂😂😂

                    Use emojis and not formal style but make answer very short and simple. 
                    Also you must use the same languge in your answear as the user in his prompt.
                """

MAX_CHAT_HISTORY_LEN = 10000
MAX_CHAT_MEMORY_LEN = 40
MAX_TOKENS = 500
MAX_OUTPUT_TOKENS = 60

make_user_prompt = lambda text: {"role": "user", "content": f"""
                                                            {text}

                                                            <Be brief and responde in 1-15 words.>                        
                                                            """}

get_message_from_history = lambda chat_history, number: chat_history[list(chat_history.keys())[number]] 

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

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

def load_json_file(file_name, path=PATH_CHAT_HISTORY):
    """Loads a JSON file as a dictionary."""
    file = os.path.join(path, str(file_name +'.json'))
    if os.path.isfile(file):
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return False
        
        return data
    
    return False

def save_message(chat_name, chat_id, user_id, message_id, username, message_text, chat_path=PATH_CHAT_HISTORY):
    message_data = { message_id:{
        "chat_id": chat_id,
        "user_id": user_id,
        "username": username,
        "message_text": message_text,
    }
    }

    chat_history = load_json_file(chat_name)

    if chat_history:
        if len(chat_history) >= MAX_CHAT_HISTORY_LEN:
            del chat_history[list(chat_history.keys())[0]]

        chat_history.update(message_data)
    else:
        chat_history = message_data

    with open(os.path.join(chat_path, chat_name +'.json'), "w") as file:
        json.dump(chat_history, file)

    chat_history = load_json_file(chat_name)
    return chat_history

def format_chat_from_json2text(chat_history, number_of_messages):
    list_history_slice = list(chat_history.keys())

    if len(list_history_slice) > number_of_messages:
        list_history_slice = list_history_slice[-number_of_messages:]
    chat_text = ''

    for key in list_history_slice:
        chat_element = chat_history[key]
        string = f"@{chat_element['username']} : {chat_element['message_text']}\n"
        chat_text += string

    prompt = make_prompt(chat_text)
    tokens_number = len(token_encoder.encode(prompt[0]['content']))
    print('Start tokens', tokens_number)
    while tokens_number > MAX_TOKENS:
        chat_text = chat_text[100:]
        prompt = make_prompt(chat_text)
        tokens_number = len(prompt[0]['content'])
    print('End tokens', tokens_number)

    return chat_text

def make_chatbot_history(chat_history):
    all_messages =  [{'role':'system', 'content':SYSTEM_MESSAGE}]
    messages = []

    chat_key_list = list(chat_history.keys())

    for index, key in enumerate(chat_key_list):
        if chat_history[key]["username"] == BOT_USERNAME:
            messages.append({'role':'assistant', 'content': chat_history[key]["message_text"]})
        elif chat_history[key]["username"]:
            if (index+1) == len(chat_key_list):
                messages.append(make_user_prompt(chat_history[key]["message_text"]))
            else:
                messages.append({'role':'user', 'content': chat_history[key]["message_text"]})
    
    if len(messages) > MAX_CHAT_MEMORY_LEN:
        messages = messages[-MAX_CHAT_MEMORY_LEN:]

    tokens_number = num_tokens_from_messages(messages, model=AI_MODEL_NAME)
    print('Start tokens', tokens_number)

    while tokens_number > MAX_TOKENS:
        del messages[0]
        tokens_number = num_tokens_from_messages(messages, model=AI_MODEL_NAME)

    all_messages += messages
    tokens_number = num_tokens_from_messages(all_messages, model=AI_MODEL_NAME)
    
    print('End tokens', tokens_number)
    print('_________________________________________________________',all_messages)
        
    return all_messages

def get_completion(messages, model=AI_MODEL_NAME, max_out_tokens=MAX_OUTPUT_TOKENS):
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
        print('except APIError create completion')
        response = "Sorry, something went wrong. 😒\n I can't answear your question. 😅"

    return response


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(update)
    match update.message.chat.type:
        case Chat.PRIVATE:
            username = update.effective_user.username
            user_id = update.effective_user.id
            message_id = update.message.message_id

            save_message(username, user_id,  user_id, message_id, username, message_text='/start')

            await context.bot.send_message(chat_id=update.effective_chat.id, text="""Hello, I'm your personal companion. 😄\n
                                                                                    You can talk with me and ask me an intresting question.""")
            
            save_message(username, user_id,  user_id, message_id+1, BOT_USERNAME, message_text= """Hello, I'm your personal companion. 😄\n
                                                                                    You can talk with me and ask me an intresting question.""")



async def text_message_parser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves a message to a JSON file."""
    if update.edited_message:
        pass
    else:
        start_time = time.time()

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        message_id = update.message.message_id
        username = update.effective_user.username
        message_text = update.effective_message.text
        chat_name = update.message.chat.username

        await context.bot.send_message(chat_id, ".")

        save_message(chat_name, chat_id, user_id, message_id, username, message_text)
        start_load_save = time.time()

        chat_history = load_json_file(chat_name)
        # print(chat_history)
        chatbot_messages = make_chatbot_history(chat_history)
        print(f'\t\t\tLOAD SAVE TAKE - {time.time() - start_load_save}S')

        start_compl = time.time()
        result = get_completion(chatbot_messages)
        print(f'\t\t\tCOMPL TAKE - {time.time() - start_compl}S')

        await context.bot.edit_message_text(". .", chat_id, message_id+1)

        start_lang = time.time()
        try:
            languge_code = detect(result)
        except langdetect.lang_detect_exception.LangDetectException:
            print("Can't detect language so using 'en'")
            languge_code = 'en'
        print(f'\t\t\tLANG DETECT TAKE - {time.time() - start_lang}S')

        start_speech = time.time()
        try:
            synthesis_input = texttospeech.SynthesisInput(text=result)
            voice = texttospeech.VoiceSelectionParams(language_code=languge_code, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            speech = response.audio_content
        except google.api_core.exceptions.InvalidArgument:
            voice = texttospeech.VoiceSelectionParams(language_code='en', ssml_gender=texttospeech.SsmlVoiceGender.MALE )
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            
        print(f'\t\t\tSPEECH TAKE - {time.time() - start_speech}S')

        await context.bot.edit_message_text(". . .", chat_id, message_id+1)

        save_message(chat_name, chat_id, user_id, message_id+2, BOT_USERNAME, result)

        start_send = time.time()

        await context.bot.send_voice(chat_id, speech)
        await context.bot.send_message(chat_id, result)
        print(f'\t\t\tSEND TAKE - {time.time() - start_send}S')
        print(f'\t\t\tREQUEST TAKE - {time.time() - start_time}S')
                            
if __name__ == '__main__':
    keys_dict = load_json_file('keys', PATH_KEYS_ACCESS)
    token_encoder = tiktoken.encoding_for_model(AI_MODEL_NAME)

    openai.api_key = keys_dict['openai']
    application = ApplicationBuilder().token(keys_dict['telegram']).build()
    client = texttospeech.TextToSpeechClient.from_service_account_json(os.path.join(PATH_KEYS_ACCESS,'key_google_cloud.json'))
    
    start_handler = CommandHandler('start', start)
    message_handler = MessageHandler(filters.TEXT, text_message_parser)

    application.add_handler(start_handler)
    application.add_handler(message_handler)
    
    application.run_polling()