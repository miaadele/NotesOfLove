import spacy

print("org:")
print(spacy.explain("ORG"))

print("\ncardinal:")
print(spacy.explain("CARDINAL"))

print("\ndate:")
print(spacy.explain("DATE"))

print("\ntime:")
print(spacy.explain("TIME"))

print("\nwork of art:")
print(spacy.explain("WORK_OF_ART"))

print("\ngpe:")
print(spacy.explain("GPE"))

print("\nperson:")
print(spacy.explain("PERSON"))