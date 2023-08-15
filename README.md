Sure, here is the README.md file for the Telegram bot that uses voice messages and sends an answer in 3 seconds:


# Speaky

Speaky is a Telegram bot that uses voice messages to communicate with people. It can answer questions, tell jokes, and even generate creative text etc.

## How to use

1. Send a text message to Speaky.
2. Speaky will process the message and respond with an answer in 3 seconds in text and voice message.
3. You can continue to ask Speaky questions or tell it to do things.

## Features

* Can answer questions in a comprehensive and informative way, even if they are open ended, challenging, or strange.
* Can tell jokes that are appropriate for all ages.

## How to get started

1. Open the Telegram app and search for the Speaky bot 'https://t.me/soundspeakerbot'.
2. Tap on the Speaky bot and then tap on the "Start" button.
3. Speaky will now be able to receive your messages.

## Commands

* /start - Starts the Speaky bot.

## Tips

* Speaky can understand a variety of accents and languages.
* If Speaky doesn't understand your question, try rephrasing it.

## Requirements

* Python 3.8 or higher
* Using following Python packages:

    * openai
    * tiktoken
    * langdetect
    * google-cloud-texttospeech
    * python-telegram-bot

## Installation

1. Install the Python packages listed in the "Requirements" section.
2. Clone the Speaky repository from GitHub:


git clone https://github.com/[username]/Speaky.git
```

3. Change directory to the Speaky directory:

```
cd Speaky
```

4. Run the following command to start the bot:

```
python3 main.py

5. Create files "key_google_cloud.json" , "keys.json" in folder 'keys_access/' and write your keys in "key_google_cloud.json" for google-cloud-texttospeech api and in "keys.json" for telegram and openai keys in next format this keys aren't working:

{ "telegram": "sdfjklsdfkjdkfjlkdsncmireojoierjgrekvle", "openai": "sk-lkjfkljdglklkdslkmslkmflmskfmsld" } 
```

The bot will now be available to answer your questions and do what you ask.
```

I hope this helps!