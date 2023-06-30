import nltk
import random

nltk.download('wordnet')
nltk.download('punkt')

def replace_word(sentence, word, replacement):
    words = nltk.word_tokenize(sentence)
    replaced_words = [replacement if w == word else w for w in words]
    replaced_sentence = ' '.join(replaced_words)
    return replaced_sentence

def generate_paraphrase(sentence):
    words = nltk.word_tokenize(sentence)
    paraphrases = []

    for i in range(len(words)):
        word = words[i]
        synonyms = nltk.corpus.wordnet.synsets(word)
        
        if len(synonyms) > 1:
            synonyms_list = []
            for syn in synonyms:
                for lemma in syn.lemmas():
                    synonyms_list.append(lemma.name().replace('_', ' '))
            if len(synonyms_list) > 1:
                ranrepl_synonym = random.choice(synonyms_list)
                para_sen = replace_word(sentence, word, ranrepl_synonym)
                paraphrases.append(para_sen)
    
    return paraphrases

#demo
sentence = str(input("Input the sentence: "))
paraphrases_gen = generate_paraphrase(sentence)
for paraphrase in paraphrases_gen:
    print(paraphrase)
