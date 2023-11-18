import re
import os
from collections import Counter
from nltk.corpus import stopwords
import nltk
import xml.dom.minidom as minidom
from collections import defaultdict
import heapq
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')


input_path = 'movie-subtitles'

movie_dialogues= {}
files_in_folder = os.listdir(input_path)

for file_name in files_in_folder:
    file_path = os.path.join(input_path, file_name)
    #print(file_name)
    # Check if the current item in the folder is a file
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:  #, encoding='utf-8'
            data = file.read()
            # Split the content into individual lines
            lines = data.split('\n')

            dialogue_content = []
            current_sentence = ''
            for line in lines:
                if re.search(r'[a-zA-Z]', line):
                    current_sentence += line + ' '
                elif not line.strip() and current_sentence:
                    dialogue_content.append(current_sentence.strip())
                    current_sentence = ''  # Reset for the next sentence
        movie_dialogues[file_name]=dialogue_content


#print(dialogue_content)
#print(movie_dialogues)
#input()

def extract_keywords(dialogue_content, num_keywords=10):

    all_dialogue = ' '.join(dialogue_content)
    #print(all_dialogue)
    # Tokenize words 
    words = re.findall(r'\b\w+\b', all_dialogue.lower())
    #print(words)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word)>4]
    #print(words)
    # Calculate word frequencies
    word_freq = Counter(words)
    # Extract most common words as potential keywords
    keywords = [word for word, _ in word_freq.most_common(num_keywords)]
    
    return keywords

def extract_keywords2(dialogue_content, num_keywords=10):

    all_dialogue = ' '.join(dialogue_content)
    words = re.findall(r'\b\w+\b', all_dialogue.lower())
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in words if word.lower() not in stop_words]
    # Calculate frequency distribution
    fdist = FreqDist(filtered_tokens)
    # Extract top n words by frequency
    top_words = fdist.most_common(num_keywords)
    
    keywords = []
    for word,n in top_words:
        keywords.append(word)

    return keywords


def extract_context(dialogue_content, keyword):
    context_sentences = []
    for i, dialogue in enumerate(dialogue_content):
        # Searching for the keyword or phrase indicating information need
        if keyword.lower() in dialogue.lower():
            #print(dialogue)
            
            start_index = max(0, i - 5)
            end_index = min(len(dialogue_content), i + 6)
            context = ' '.join(dialogue_content[start_index:end_index])
            context_sentences.append(context)
            break
            
    return context_sentences


import xml.etree.ElementTree as ET

def prettify(elem):
    # Return a pretty-printed XML string representation
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")  

def text_to_xml(root, input_text_file, output_xml_file, num_txt, title_txt, movie_txt):
    
    # Creating XML structure

    top = ET.SubElement(root, 'top')

    num = ET.SubElement(top, 'num')
    num.text = num_txt  # Assuming the topic number is fixed

    title = ET.SubElement(top, 'title')
    title.text = title_txt  # Assuming the title is fixed

    movie = ET.SubElement(top, 'movie')
    movie.text = movie_txt  # Assuming the movie name is fixed

    desc = ET.SubElement(top, 'desc')  # Description element

    lines = nltk.sent_tokenize(input_text_file[0])
    # Adding <p> elements for each line in the text file
    for line in lines:
        p = ET.SubElement(desc, 'p')
        p.text = line.strip()  # Assigning text content to <p> elements

    # Creating XML tree
    tree = ET.ElementTree(root)

    xml_string = prettify(root)

    # Writing prettified XML to a file
    with open(output_xml_file, 'w', encoding='utf-8') as file:
        file.write(xml_string)



root = ET.Element('topics')  # Root element
num=0

for movie in movie_dialogues.keys():
    dialogue_content = movie_dialogues[movie]
    potential_keywords = extract_keywords2(dialogue_content)
    print(potential_keywords)
    
    for keyword in potential_keywords:
        num+=1
        num_txt = str(num)
        title = keyword
        movie_name = movie.split('.English')[0].replace('.', ' ')
        relevant_context = extract_context(dialogue_content, keyword)
        #print(relevant_context)
        text_to_xml(root, relevant_context, 'output_data.xml',num_txt, title, movie_name)