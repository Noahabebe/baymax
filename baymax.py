import platform
import socket
import subprocess
import time
from groq import Groq, GroqError
import keyboard
from openwakeword.model import Model
from openwakeword.vad import VAD
import pyaudio
import numpy as np
from textblob import TextBlob
import tqdm
import whisper
import speech_recognition as sr
import threading
import os
import json
import psycopg2
import bcrypt
import secrets
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pyttsx3
from nltk.metrics import edit_distance
import nltk
import pyaudio
from nltk.sentiment.vader import SentimentIntensityAnalyzer

os.environ['GROQ_API_KEY'] = "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"

whisper_model = whisper.load_model("base")
tts_engine = pyttsx3.init()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("The GROQ_API_KEY environment variable is not set.")


def load_user_api_key(file_path='user_api_key.txt'):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def load_email_api_key(email):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT api_key FROM users WHERE email = %s', (email,))
    api_key = cursor.fetchone()
    cursor.close()
    conn.close()
    return api_key[0] if api_key else None


def save_user_api_key(api_key, file_path='user_api_key.txt'):
    if api_key is not None:
        with open(file_path, 'w') as file:
            file.write(api_key)
    else:
        print("Error: API key is None, cannot save.")


def user_auth():
    while True:
        email = input('Enter your email: ')
        password = input('Enter your password: ')
       
        if validate_user(email, password):
            print("Successfully signed in.")
            user_api_key = load_email_api_key(email)
            save_user_api_key(user_api_key)
            return user_api_key, email
        else:
            print("Invalid email or password.")
            register = input("Would you like to register with a new email and password? (yes/no): ")
            if register.lower() == "yes":
                email = input("Enter your email: ")
                password = input("Enter your password: ")
                user_api_key = generate_api_key()
                store_user(email, password, user_api_key)
                print(f'Your account has been registered. Your API key is: {user_api_key}')
                save_user_api_key(user_api_key)
                
                return user_api_key,email
            else:
                print("Please try again.")

def validate_user(email, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE email = %s', (email,))
    stored_password = cursor.fetchone()
    cursor.close()
    conn.close()
    if stored_password:
        stored_password_hash = stored_password[0].encode('utf-8')  
        return bcrypt.checkpw(password.encode('utf-8'), stored_password_hash)
    else:
        return False
    

client = Groq(api_key=api_key)


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def connect_db():
    conn = psycopg2.connect(
        dbname='baymax',
        user='postgres',
        password='Aby09116393',
        host='0.tcp.ngrok.io',
        port='17249'
    )
    return conn

def generate_api_key():
    return secrets.token_urlsafe(32)

def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

def store_user(email, password, user_api_key):
    conn = connect_db()
    cursor = conn.cursor()
    hashed_password = hash_password(password).decode('utf8')
    cursor.execute(
        'INSERT INTO users (email, password, api_key) VALUES (%s, %s, %s)',
        (email,hashed_password, user_api_key)
    )
    conn.commit()
    cursor.close()
    conn.close()

def delete_api_key(email, file_path='user_api_key.txt'):
    try:
        os.remove(file_path)
        print("API key file removed successfully for the user with email:", email)
    except FileNotFoundError:
        print("API key file not found.")
    


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

        return best_match if best_score >= 0.5 else None

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
                "content": f"I'm running commands on {system_info['System Type']} . Only provide the specified command from the user prompt for this system. My computer specification is Username: {system_info['Username']}, Computer Name: {system_info['Computer Name']}, IP Address: {system_info['IP Address']}, System Type: {system_info['System Type']}. Wrap all commands in these quotes `` "
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
    
    speak(response)

    return response



def speak(text):
    engine = pyttsx3.init(driverName='sapi5')
    engine.say(text)
    engine.runAndWait()

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
    completion = client.chat.completions.create(
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
        stream=True,
        stop=None,
    )
    
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
        print(chunk.choices[0].delta.content or "", end="")
    speak(response)    
    return response

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
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(user_input)
    return sentiment_scores['compound']



def get_user_email_from_api_key(api_key):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT email FROM users WHERE api_key = %s', (api_key,))
    user_email = cursor.fetchone()
    cursor.close()
    return user_email[0] if user_email else None

def get_stored_password(email):
    conn = connect_db()
    if conn:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT password FROM users WHERE email = %s', (email,))
                stored_password = cursor.fetchone()
                return stored_password[0] if stored_password else None
    else:
        return None
    
def get_audio_data(stream):
    try:
        audio_data = stream.read(1280, exception_on_overflow=False)
        return np.frombuffer(audio_data, dtype=np.int16)  
    except IOError:
        return None


def recognize_speech_or_text():
    user_api_key = load_user_api_key()  
    
    email = get_user_email_from_api_key(user_api_key)


    command_manager = CommandManager('commands.json')
    history = []
    pending_command_storage = {}

    # Initialize OpenWakeWord
    model = Model(
        wakeword_models=["C:/Users/noahs/Downloads/baymax/baymax/baymax.onnx"],
        vad_threshold=0.5,
        inference_framework='onnx'
    )

    vad = VAD()

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
    
    while True:
        print('\n Listening for the wake word "Baymax"... or press any key')
        
        if keyboard.read_key() :
            user_input = input("Baymax: ")
        else: 
            try:
                last_hotword_time = 0

                while True:
                    audio_frame = get_audio_data(stream)
                    if audio_frame is None:
                        continue
                    
                    # Get predictions for the frame
                    prediction = model.predict(audio_frame)
                    
                    if prediction["baymax"] > 0.3:
                        current_time = time.time()
                        if current_time - last_hotword_time > 3:  # Check if 3 seconds have passed since last hotword detection
                            last_hotword_time = current_time
                            print("Please say something:")
                            
                            audio_data = []
                            vad_silence_count = 0
                            
                            while True:
                                frame = get_audio_data(stream)
                                if frame is None:
                                    continue
                                
                                audio_data.append(frame)  
                                vad_prediction = vad.predict(frame)
                                
                                if vad_prediction < 1:
                                    vad_silence_count += 1
                                    if vad_silence_count > 50:
                                        break
                                else:
                                    vad_silence_count = 0
                            
                            audio_array = np.concatenate(audio_data)  
                            audio_array = audio_array.astype(np.float32) / 32768.0  
                            
                            
                            result = whisper_model.transcribe(audio_array)
                            
                            user_input = result["text"]

            except Exception as e:
                print(f"An error occurred: {e}")

            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
        if (user_input != 'you' or user_input != 'you'):
            print("You:", user_input)

        # Process the detected command (as in your original code)
        if user_input.lower() == "exit":
            return False

        if user_input.lower() == "sign out":
            delete_api_key(email, file_path='user_api_key.txt')
            return False

        if user_input.lower() == "show history":
            show_history(history)
            print("History shown")

        if pending_command_storage.get("command"):
            sentiment = analyze_sentiment(user_input)
            if sentiment >= 0.0 or user_input.lower() == "yes":
                result = execute_command(pending_command_storage["command"])
                print(result)
                history.append({"role": "system", "content": str(result)})
                pending_command_storage.clear() 
            else:
                print("Command execution canceled due to negative sentiment or user input.")
                pending_command_storage.clear()  
                print("Command execution canceled")

        matching_command = command_manager.find_command(user_input)
        system_info = command_manager.get_system_info()
        print("System Info:", system_info)  
        if matching_command:
            detailed_command = construct_command(matching_command, user_input, system_info)
            print(f"Constructed Command: {detailed_command}\nWould you like me to execute this command for you? (yes/no)")
            speak("Would you like me to execute this command for you")
            pending_command_storage["command"] = detailed_command  
            history.append({"role": "user", "content": user_input})
            history.append({"role": "system", "content": f"Constructed Command: {detailed_command}\nWould you like me to execute this command for you? (yes/no)"})
        else:
            response = get_assistant_response(user_input, history, system_info)
            if response:
                extracted_command = extract_command(response)
                if extracted_command:
                    print(f"Extracted Command: {extracted_command}\nWould you like me to execute this command for you? (yes/no)")
                    speak("Would you like me to execute this command for you")
                    pending_command_storage["command"] = extracted_command  
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "system", "content": f"Extracted Command: {extracted_command}\nWould you like me to execute this command for you? (yes/no)"})
        last_hotword_time = time.time()




def main():
    user_api_key = load_user_api_key()  
    if not user_api_key:
        user_api_key, email = user_auth()  
        if not user_api_key:
            print("Error loading API key")
            return
    else:
        email = get_user_email_from_api_key(user_api_key)

    if not user_api_key:
        print("Error loading API key")
        return
    elif not email:
        print("Error loading email")    
        return

    print("Signed-in user email:", email)
    print("Welcome to Baymax!")
    print("Options: sign out, exit and show history. All options work with no spaces in front.")
    recognize_speech_or_text()
    


if __name__ == "__main__":
    main()
