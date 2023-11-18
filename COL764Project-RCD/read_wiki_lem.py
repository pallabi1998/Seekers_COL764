wiki_path = "/scratch/cse/msr/csy227517/IR/Project/final_out"


import re
import string
import math
import psutil
from rank_bm25 import BM25
from rank_bm25 import BM25Okapi
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def remove_punctuation(text):
    text = text.replace('|' , " or ")
    return re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)



def remove_tags(input_string):
    clean1 = re.compile('<.*?>')
    clean2 = re.compile(r'{{.*?}}')
    with_c = re.sub(clean1, '', input_string)
    return re.sub(clean2, '', with_c)


with open(wiki_path, 'r', encoding='utf-8') as file:
    first_lines = [file.readline() for _ in range(2000)]

#print(first_lines)


out_path = f"/scratch/cse/msr/csy227517/IR/Project/wikircd/sample_{3}"
sentences = first_lines#[50:100]
with open(out_path, "w", encoding="utf-8") as file:
    # Write each sentence to the file
    for sentence in sentences:
        file.write(sentence + "\n")

# import mwxml

# # Path to your MediaWiki XML dump file
# dump_file_path = "/scratch/cse/msr/csy227517/IR/Project/final_out"

# # Define a function to process page content
# def process_page(page):
#     # Extract the title and content of the page
#     title = page.title
#     content = page.text

#     # Print the title and content
#     print("Title:", title)
#     print("Content:", content)
#     print("=" * 40)

# # Open and process the MediaWiki dump file
# with mwxml.Dump.from_file(dump_file_path) as dump:

#     for i,page in enumerate(dump):
#         # Check if the page is a content page (not a talk page, user page, etc.)
#         if page.namespace == 0 and i<10:
#             process_page(page)

#         if i> 10:
#             break


# import xml
# import xml.etree.ElementTree as ET

# tree = ET.parse(wiki_path)
# root = tree.getroot()


# def extract_title_and_text(page_element):
#     title_element = page_element.find(".//title")
#     text_element = page_element.find(".//text")
    
#     if title_element is not None and text_element is not None:
#         title = title_element.text
#         text = text_element.text
#         return title, text
#     else:
#         return None, None

# # Iterate through the page elements in the XML
# for i, page_element in enumerate(root.findall(".//page")):
#     if i < 10:
#         title, text = extract_title_and_text(page_element)
        
#         if title is not None and text is not None:
#             print("Title:", title)
#             print("Text:", text)
#             print("=" * 40)

#     else:
#         break

from lxml import etree

# Path to your MediaWiki XML export file


# Create an ElementTree and a parser object
parser = etree.XMLParser(recover=True)

# Define a function to extract title and text from a page element
def extract_title_and_text_(page_element):
    # children = page_element.getchildren()
    # for child in children:
    #     print(child.keys, child.tag)
    #print(page_element.tag)
    title_element = page_element.find(".//title")
    text_element = page_element.find(".//text")
    #print(title_element, text_element)
    
    if title_element is not None and text_element is not None:
        title = title_element.text
        text = text_element.text
        return title, text
    else:
        children = page_element.getchildren()
        for child in children:
            if child.tag[-1] == 'p':
                sub_children = child.getchildren()
                for sc in sub_children:
                    print(sc.text, sc.tag)


        return None, None

# Iterate through the page elements in the XML
# with open(wiki_path, 'rb') as f:
#     cnt = 0
#     for event, elem in etree.iterparse(f, events=('start', 'end'), recover=True):
        
#         if 'page' in elem.tag and event == 'end':
            
#             cnt += 1
#             title, text = extract_title_and_text(elem)
            
#             if title is not None and text is not None:
#                 print("Title:", title)
#                 print("Text:", text)
#                 print("=" * 40)
            
#             # Clear the element to release memory
#             elem.clear()
#         else:
#             #print(elem.tag, event)
#             pass
#         if cnt == 15:
#             break





#passages_list = []
#for passage in passages:
def process_passage(passage):
    lines = passage.split("\n")
    rem = ""
    title = ""
    whole = True
    add_rem = False
    for line in lines:
        if not add_rem:
            if "<pno>" in line:
                pno = line[len("<pno>") : -len("</pno>")]
                #print(pno)
            else: 
                if '<title>' in line:
                    whole = False
                    title = line[len("<title>") : -len("</title>")]
                
                if ('<text' in line):
                    whole = False
                    add_rem = True
                    rem += remove_tags(line) + " "
        else:
            rem += remove_tags(line) + " "


    if whole:
        for line in lines:
            if ("<pno>" not in line) and ("<sha" not in line):
                rem += remove_tags(line) + " "

    if len(rem.strip()) > 0:
        return [pno.strip(), title.strip(), rem.strip()]
        #passages_list.append([pno.strip(), title.strip(), rem.strip()])


#print(len(passages_list))


def find_titles_in_file(file_path, max_titles=3, start=0):
    in_p_tag = False
    passages_no = []
    passages_text = []
    LCNT = 0
    with open(file_path, 'r') as file:
        for line in file:
            LCNT += 1
            if LCNT <= start:
                continue
            line = line.strip()

            if in_p_tag:
                p += line + "\n"
                if "</p>" in line:
                    in_p_tag = False
                    pro_p = process_passage(p)
                    if pro_p is not None:
                        passages_no.append(pro_p[0])
                        passages_text.append(pro_p[1]+remove_punctuation(pro_p[2]))
                    if len(passages_no) >= max_titles:
                        break
                    if len(passages_no) % 20000000 == 0 and len(passages_no) > 0:
                        print(len(passages_no), passages_text[-1], passages_no[-1])
                        print(psutil.virtual_memory().available)
                        #if len(passages_no) % 50000000 == 0 or psutil.virtual_memory().available < 1000000000 :
                        tokenized_corpus = []
                        for doc in passages_text:
                            temp_l = doc.split(" ")                     
                            for idx, word in enumerate(temp_l):
                                temp_l[idx] = lemmatizer.lemmatize(word)
                            tokenized_corpus.append(temp_l)
                        
                        bm25 = BM25Okapi(tokenized_corpus)
                        with open(f"bm25_lem_model_{LCNT}.pkl", "wb") as f:
                            pickle.dump(bm25, f)
                        del bm25
                        del tokenized_corpus
                        del passages_no
                        del passages_text
                        passages_text = []
                        passages_no = []
                        print("loaded")

                        

            if "<p>" in line:
                p = ""
                in_p_tag = True

            else:
                pass 
                #print(f"trashing : {line}")
            

    return passages_no, passages_text



passages_no, passages_text = find_titles_in_file(wiki_path, max_titles=math.inf, start=229868210)
tokenized_corpus = [doc.split(" ") for doc in passages_text]
bm25 = BM25Okapi(tokenized_corpus)
with open(f"bm25_model_last.pkl", "wb") as f:
    pickle.dump(bm25, f)
print("loaded")
# for i, title in enumerate(passages, start=1):
#     print(f"Title {i}: {title}")



# for p in passages_list:
#     print(remove_punctuation(p[2])) 


"""

ELEM :


['__bool__', '__class__', '__contains__', '__copy__', 
'__deepcopy__', '__delattr__', '__delitem__', '__dir__', 
'__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', 
'__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__ne__',
 '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__setattr__', '__setitem__',
  '__sizeof__', '__str__', '__subclasshook__', '_init', 'addnext', 'addprevious', 'append', 'attrib',
   'base', 'clear', 'cssselect', 'extend', 'find', 'findall', 'findtext', 'get', 'getchildren',
    'getiterator', 'getnext', 'getparent', 'getprevious', 'getroottree', 'index', 'insert', 
    'items', 'iter', 'iterancestors', 'iterchildren', 'iterdescendants', 'iterfind', 'itersiblings', 
    'itertext', 'keys', 'makeelement', 'nsmap', 'prefix', 'remove', 'replace', 'set', 'sourceline', 
    'tag', 'tail', 'text', 'values', 'xpath']



CHILD :
    ['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', 
    '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', 
    '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', 
    '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', 
    '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 
    'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
"""