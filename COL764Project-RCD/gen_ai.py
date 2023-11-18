import google.generativeai as palm

api_key = "AIzaSyAI5jkKel_XPxiVgwKIOSgHaKXvyYNtgdY"
palm.configure(api_key=api_key)

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods] #if 'generateText' in m.supported_generation_methods
model = models[0].name
print(models)
print(model)



inp_file = "./RCD2020FIRETASK/trec_formatted_with_p_tags.txt"


import xml
import xml.etree.ElementTree as ET

tree = ET.parse(inp_file)
root = tree.getroot()



    
titles = []
texts = []
answer = []

for top_element in root.findall("top"):
    #print(top_element)
    title_element = top_element.find("title")
    desc_element = top_element.find("desc")
    text_element = ""
    for p_elem in desc_element.findall("p"):
        text_element += p_elem.text + "\n"
    titles.append(title_element.text)
    texts.append(text_element.strip())

cnt = 25
retry = 0
while cnt < len(texts):
    text = texts[cnt]



    """ as shown in the examples below
###

Dialogue : 
All right. It's not Sunday. We don't need a sermon.
to be able to behave like gentlemen. 
it. 
close the window. It was blowing on my neck. 
t seems to me that it's up to us to convince this gentleman (indicating NO. 8) that we're right and he's wrong. Maybe if we each took a minute or two, you know, if we sort of try it on for size.
table. 
guilty. I thought it was obvious. I mean nobody proved otherwise.is on the prosecution. The defendant doesn't have to open his mouth. That's in the Constitution. The Fifth Amendment. You've heard of it. 
I . . . what I meant . . . well, anyway, I think he was guilty. 
man who lived on the second floor right underneath the room where the murder took place. At ten minutes after twelve on the night of the killing he heard loud noises in the upstairs apartment. He said it sounded like a fight. Then he heard the kid say to his father, "I'm gonna kill you.!” A second later he heard a body falling, and he ran to the door of his apartment, looked out, and saw the kid running down the stairs and out of the house. Then he called the police. They found the father with a knife in his chest. 
movies. That's a little ridiculous, isn't it? He couldn't even remember what pictures he saw. 
right. 
testimony don't prove it, then nothing does. 

Difficult: Fifth Amendment
###

Dialogue:
What about the ones that were asked? For instance, let's talk about that cute little switchknife. You know, the one that fine, upright kid admitted buying.
at it. I'd like to see it again, Mr. Foreman.
look at it again. What do yo: We all know what it looks like. I don't see why we have to look at it again. What do you think? 
don't you agree? 
being slapped by his father.
switch knife. The storekeeper was arrested the following day when he admitted selling it to the boy. It's a very unusual knife. The storekeeper identified it and said it was the only one of its kind he had in stock. Why did the boy get it? As a present for a friend of his, he says. Am I right so far?
what he's talking about. 
fallen through a hole in his coat pocket, that he never saw it again. Now there's a story, gentlemen. You know what actually happened. The boy took the knife home and a few hours later stabbed his father with it and even remembered to wipe off the fingerprints.
are you trying to tell me that someone picked it up off the street and went up to the boy's house and stabbed his father with it just to be amusing?
and that someone else stabbed his father with a similar knife. It's possible.
never seen one like it before in my life and neither had the storekeeper who sold it to him.

Difficult: switch knife
###

"""
# Give short summary from Dialogue:
    prompt= f""" Find one or two Difficult words from Dialogue:


Dialogue: {text}

Difficult words are: """
    
    #prompt ="hello this is a story about a king and "

    # DEFAULT_ARGS = {
    #         'model': 'models/text-bison-001',
    #         'max_output_tokens': 256,
    #         'temperature': 0.5
    #     }

    #completion = palm.generate_text(**{'prompt': prompt, **DEFAULT_ARGS})
    try:

        completion = palm.generate_text(
            model='models/text-bison-001',
            prompt=prompt,
            temperature=0.5,
            # The maximum length of the response
            max_output_tokens=4,
        )
        if completion.result is not None or retry >= 0:
            
            answer.append(completion.result)
            print(completion.result, completion)
            cnt += 1
            retry = 0
        else: 
            retry += 1


    except Exception:
        print(Exception)



with open("qnswer_file_palm_2", "w") as f:
    for a in answer:
        if a is not None:
            a.replace('\n', ' ')
            f.write(a+'\n')
        else:
            f.write(" \n")

