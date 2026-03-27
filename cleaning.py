import json
import re

#punctuation marks that will be kept
KEEP_PUNCT = {"'", ",", ".", "?", "!"}

#custom list of filler words to remove
FILLER_WORDS = {"Ah", "ah", "Ahh", "ahh", "Ay", "ay", "Ba", "ba", "Bah", "bah", "Boaw", "boaw", "Cha", "cha", "Choo", "choo", "Da", "da", "De", "de", "Dee", "dee", "Dodo", "dodo", "Dododo", "dododo", "Doo", "doo", "Dooby", "dooby","Du", "du", "Dum", "dum", "Eh", "eh", "ehhhhh", "ehhhhhhhh", "ehhhhhh", "Embed", "Hehhhh", "hehhhh", "Hey", "hey", "Hm", "hm", "Hmm", "hmm", 'Hmmmmm', "hmmmmm", "Hey", "hey", "Ho", "ho", "Hoo", "hoo", "Hooh", "hooh", "Huh", "huh", "Huhhh", "huhhh", "La", "la", "mama-sa", "ma-ma-ko-ssa", "Mama-say", "Mm", "mm", "Mmm", "mmm", "Mmmm", "mmmm", "Mmmmm", "mmmmmm", "Na", "na", "Nah", "nah", "Oh", "oh", "Ooh", "ooh", "Ohh", "ohh", "Oohhh", "oohhh", "Oohh", "oohh", "Ohhh", "ohhh", "Oooo", "oooo", 'Ohooohooohhh', "Ooohooohooohooo", "Ooooooh", "ooooooh", "Oooooohh", "Ohoho", "ohoho", "Ow", "ow", "Ron", "ron", "Shoop", "shoop", "Shoo", "shoo", "Uh", "uh", "Uhh", "uhh", "Uhm", "uhm", "Whoa", "whoa", "Whooa", "whooa", 'Whoooooa', "whoooooa", "Woo", "woo", "Wooo", "wooo", "Woooo", "woooo", "Ya", "ya", "Yea", "yea", "Yeah", "yeah", "Yeahh", 'yeahh'}

def normalize_lyrics(text):
    if not text:
        return ""
    
    #remove bracketed sections like [Chorus 1], [Verse 2]
    text = re.sub(r"\[.*?\]", "", text)

    #remove sections in parenthesis 
    text = re.sub(r"\(.*?\)", "", text)

    #remove sections in braces 
    text = re.sub(r"\{.*?\}", "", text)

    #remove unwanted punctuation
    text = "".join(
        ch if ch.isalnum() or ch.isspace() or ch in KEEP_PUNCT else " "
        for ch in text
    )

    #tokenize
    words = text.split()

    #remove filler words
    words = [w for w in words if w not in FILLER_WORDS]

    #rejoin 
    return " ".join(words)

#load filtered data
with open("filtered.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#apply normalization
for entry in data:
    entry["Lyrics"] = normalize_lyrics(entry.get("Lyrics"))

#save new file
with open("cleaned_lyrics.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent = 2, ensure_ascii= False)