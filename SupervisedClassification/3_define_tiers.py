# The lyrics with "love" are divided into 4 clusters, 
# so I am going to build a classifier to distinguish between 4 uses of the concept 'love.'
#Each concept will have a tiered list of A and B
# A: CORE triggers that explicitly refer to the specific context of love and 
# B: MAYBE triggers that are adjacent to the context of love

#CONTEXT 1: romantic love
ROM_A = {
    "lover", "loving", "make love",
    "be mine", "hold me", "hold you",
    "kiss", "kissing",
    "girl","baby", "honey",
    "sex", "heart"
}
ROM_B = {
    "madly", "deeply",
    "sweetness", "sweetly", "affection",
    "satisfy", "pleasing",
    "attracted", "dearest",
    "cuddle", "huggin",
    "desires", "temptation", 
    "wife", "husband",
    "lips", "bed"
}

#CONTEXT 2: longing and uncertainty
LONGING_A = {
    "bittersweet", "forgiving", "miss", "missing",
    "desired", "used to be",
    "yearning", "grieve", "pleading",
    "fragile", "carefully", "helplessly"
}
LONGING_B = {
    "dies", "unreal",
    "disaster", 
    "deepest", "haunt",
    "deceived", "discouraged",
    "tragedy", "ashamed", "fail",
    "apologies"
}

#CONTEXT 3: time-based love
TIME_A = {
    "time",
    "tonight", "night",
    "forever", "always", "eternity", "eternally", "indefinitely", "endlessly",
    "lifetime", "lifetimes"
    "today","tomorrow",
    "year", "years", "month", "months", "day", "days", "decade", "decades"
}
TIME_B = {
    "life",
    "chapter",
    "vow", "marriage",
    "lasts", "devotion", "keeping"
}
#CONTEXT 4: hopeful
HOPE_A = {
    "conquer", "guide", "achieve",
    "blessings", "miracles", "miracle",
    "believes", "belongs",
    "fulfill", "heals"
}
HOPE_B = {
    "protection", "courage", "remedy",
    "sincere",
    "repair", "soothe", 
    "cheerleader",
    "sacred", "sunrise",
    "soar"
}