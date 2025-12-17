import pyttsx3 as pyt
import engineio as eng

eng = pyt.init()

voices = eng.getProperty('voices')
eng.setProperty('rate', 150)
eng.setProperty('voice', voices[0].id)

def speak(text):
    eng.say(text)
    eng.runAndWait()

speak('what do you want me to say?')

user_input = input('Enter the text you want me to say: ')
speak(user_input)
