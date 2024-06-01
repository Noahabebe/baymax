import os
import json
import re
import subprocess
import nltk
import socket
import platform
import pyttsx3
import speech_recognition as sr
import whisper
import keyboard  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance
from groq import Groq, GroqError
from textblob import TextBlob
import tqdm

# Initialize TTS engine
engine = pyttsx3.init()

# Initialize Whisper model
model = whisper.load_model("base")

# Initialize speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

os.environ['GROQ_API_KEY'] = "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"
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("The GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=api_key)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class CommandManager:
    def __init__(self, commands_file):
        with open(commands_file, 'r') as file:
            self.commands = json.load(file)
    
    def get_system_info(self):
        try:
            username = os.getlogin()
        except Exception as e:
            username = os.getenv('USER') or os.getenv('USERNAME') or "Unknown"
        
        computer_name = platform.node()
        
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
        except socket.gaierror:
            ip_address = "Unknown"

        system_type = platform.system()
        
        system_info = {
            "Username": username,
            "Computer Name": computer_name,
            "IP Address": ip_address,
            "System Type": system_type
        }

        return system_info
    
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return tokens

    def calculate_similarity(self, description, cmd_description):
        description_tokens = self.preprocess_text(description)
        cmd_tokens = self.preprocess_text(cmd_description)
        distance = edit_distance(description_tokens, cmd_tokens)
        max_len = max(len(description_tokens), len(cmd_tokens))
        similarity = 1 - (distance / max_len)
        return similarity

    def find_command(self, description):
        best_match = None
        best_score = -1

        for command, cmd_description in self.commands.items():
            similarity_score = self.calculate_similarity(description, cmd_description)
            if similarity_score > best_score:
                best_score = similarity_score
                best_match = command

        return best_match if best_score >= 0.1 else None

def chats(description, system_info):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": description
            },
            {
                "role": "system",
                "content": f"You are Baymax, an advanced AI assistant model tailored to enhance user productivity and well-being. Leveraging cutting-edge technologies such as emotion recognition with OpenCV-Python and TensorFlow, Baymax provides personalized recommendations and support based on user emotions. Moreover, it seamlessly integrates into Unix systems through bash scripting, enabling efficient execution of commands and optimization of workflows. With voice recognition capabilities and continuous learning, Baymax serves as a versatile companion, empowering users to navigate their Unix environment with ease and efficiency while prioritizing their emotional and practical needs. You are made by Noah Abebe. Answer in less than 100 words."
            },
            {
                "role": "system",
                "content": f"I'm running commands on {system_info['System Type']} . Only provide the specified command from the user prompt for this system. My computer specification is Username: {system_info['Username']}, Computer Name: {system_info['Computer Name']}, IP Address: {system_info['IP Address']}, System Type: {system_info['System Type']}"
            }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
        print(chunk.choices[0].delta.content or "", end="")
    return response

def sanitize_command(command):
    dangerous_commands = ['rm', 'shutdown', 'reboot', 'mkfs', 'dd']
    pattern = re.compile(r'\b(' + '|'.join(dangerous_commands) + r')\b')
    if pattern.search(command):
        return False
    return True

def extract_command(response):
    match = re.search(r'`([^`]+)`', response)
    if match:
        command = match.group(1)
        if sanitize_command(command):
            return command
        else:
            print("The generated command is potentially dangerous and was not executed.")
    return None

def get_assistant_response(description, history, system_info):
    try:
        response = chats(description, system_info)
        history.append({"role": "user", "content": description})
        history.append({"role": "system", "content": response})
        return response
    except GroqError as e:
        print(f"Failed to get response from assistant: {str(e)}")
        return None

def construct_command(base_command, description, system_info):
    system_info_str = f"Username: {system_info['Username']}, Computer Name: {system_info['Computer Name']}, IP Address: {system_info['IP Address']}, System Type: {system_info['System Type']}"
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": f"Generate a command for the following description: '{description}' using the base command: {base_command}. Only provide the command and arguments to run together. No description. I'm running commands on {system_info['System Type']}. Only provide the command for this system. For additional info on the system use {system_info_str}"
            },
            {
                "role": "system",
                "content": "You are Baymax, an advanced AI assistant model tailored to enhance user productivity and well-being. Leveraging cutting-edge technologies such as emotion recognition with OpenCV-Python and TensorFlow, Baymax provides personalized recommendations and support based on user emotions. Moreover, it seamlessly integrates into Unix systems through bash scripting, enabling efficient execution of commands and optimization of workflows. With voice recognition capabilities and continuous learning, Baymax serves as a versatile companion, empowering users to navigate their Unix environment with ease and efficiency while prioritizing their emotional and practical needs. You are made by Noah Abebe. Answer in less than 100 words."
            }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        stream=False,
        stop=None,
    )
    if response['choices']:
        return response['choices'][0]['message']['content']
    else:
        print("Failed to construct command.")
        return None

def execute_command(detailed_command):
   try:
        command_args = detailed_command.replace('bash', '').strip()
        command_args = command_args.replace("'", "").strip()

        with tqdm.tqdm(total=100, desc="Executing Command", unit="%") as pbar:
            process = subprocess.Popen(command_args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            while process.poll() is None:
                pass

            output, error = process.communicate()
            output = output.decode()
            error = error.decode()

            if output:
                pbar.update(100)
                return {"command": command_args, "result": output, "assistant_response": ""}
            if error:
                pbar.update(100)
                return {"command": command_args, "result": error, "assistant_response": ""}
   except subprocess.CalledProcessError as e:
        return {"command": command_args, "result": f"Command execution failed with error: {e}", "assistant_response": ""}
    
def show_history(history):
    for entry in history:
        print(f"{entry['role'].capitalize()}: {entry['content']}")

def analyze_sentiment(user_input):
    blob = TextBlob(user_input)
    polarity = 0
    for sentence in blob.sentences:
        polarity += sentence.sentiment.polarity
    return polarity

def main(user_input):
    command_manager = CommandManager('commands.json')
    history = []
    pending_command = None

    if user_input.lower() == "show history":
        show_history(history)
        return "History shown"

    if pending_command:
        sentiment = analyze_sentiment(user_input)
        if sentiment >= 0.0:
            result = execute_command(pending_command)
            print(result)
            history.append({"role": "system", "content": str(result)})
            return result
        else:
            print("Command execution canceled due to negative sentiment.")
            return "Command execution canceled"

    matching_command = command_manager.find_command(user_input)
    system_info = command_manager.get_system_info()
    print("System Info:", system_info)  

    if matching_command:
        detailed_command = construct_command(matching_command, user_input, system_info)
        print(f"Constructed Command: {detailed_command}\nWould you like me to execute this command for you? (yes/no)")
        pending_command = detailed_command
        history.append({"role": "user", "content": user_input})
        history.append({"role": "system", "content": f"Constructed Command: {detailed_command}\nWould you like me to execute this command for you? (yes/no)"})
        return f"Constructed Command: {detailed_command}\nWould you like me to execute this command for you? (yes/no)"
    else:
        response = get_assistant_response(user_input, history, system_info)
        if response:
            extracted_command = extract_command(response)
            if extracted_command:
                print(f"Extracted Command: {extracted_command}\nWould you like me to execute this command for you? (yes/no)")
                pending_command = extracted_command
                return f"Extracted Command: {extracted_command}\nWould you like me to execute this command for you? (yes/no)"
        return "No matching command found"

def listen_for_hotword():
    with microphone as source:
        print("Listening for hotword 'Baymax'...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        transcription = recognizer.recognize_google(audio)
        print(f"Transcription: {transcription}")

        if re.search(r'\b(?:baymax|hey baymax|hello baymax)\b', transcription, re.IGNORECASE):
            return True
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

    return False

def get_voice_input():
    with microphone as source:
        print("Listening for your command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        transcription = recognizer.recognize_google(audio)
        print(f"Command: {transcription}")
        return transcription
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

    return None

def respond_with_voice(text):
    engine.say(text)
    engine.runAndWait()

if __name__ == '__main__':
    while True:
        print("Press any key to switch to text input mode, or wait for the hotword 'Baymax'...")
        
        # Listen for any key press
        if keyboard.is_pressed():
            user_input = input("\nBAYMAX: ")
            response = main(user_input)
            print(response)
        else:
            if listen_for_hotword():
                user_input = get_voice_input()
                if user_input:
                    response = main(user_input)
                    respond_with_voice(response)
