import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalnum()]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Split the input when encountering a comma
    sentences = sentence.split(',')
    return_list = []
    for sentence in sentences:
        sentence = sentence.strip()  # Remove leading/trailing spaces
        sentence = sentence.replace(',', ' ')
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
    
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't quite catch that. Can you please provide more details or ask another question?"

    responses = {}
    for intent in intents_json['intents']:
        tags = intent['tag'].split(',')
        for tag in tags:
            responses[tag.strip()] = intent.get('responses', [])

    result = []
    unique_tags = set()  # To store unique tags encountered
    for intent_data in intents_list:
        tags = [tag.strip() for tag in intent_data['intent'].strip().split(',')]
        for tag in tags:
            if tag in responses and tag not in unique_tags:
                unique_tags.add(tag)
                tag_responses = responses[tag]
                # Select a random response for the tag
                random_response = random.choice(tag_responses)
                result.append(random_response)
    return '\n'.join(result) if result else "No responses found for provided tags."



print("GO! Bot is running!")

while True:
    user_input = input("You: ")
    ints = predict_class(user_input)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
