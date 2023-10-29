detector = Module()
language = 'en'
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Please say something....")
        audio = recognizer.listen(source, timeout=2)
        try:
            print("You said: \n" + recognizer.recognize_google(audio))
            return (recognizer.recognize_google(audio))
        except Exception as e:
            print("Error: " + str(e))
