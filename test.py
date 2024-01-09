import os
import cv2
import openai
import json
import requests
from google.cloud import translate_v2 as translate
from google.cloud import vision
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


"""
Initializes local llms through generator and initializes the tokenizer
"""
def init_llm():
    model_name_or_path = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
    # To use a different branch, change revision
    # For example: revision="latest"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="latest")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

"""
Initializes openai by importing key from OpenAIKey file
"""
def init_openai():
    with open("OpenAIKey", "r") as file:
        openai.api_key = file.read()

"""
Detects text in file (path) using google vision api.
Google vision api call returns a text annotation object from which we extract "texts" and "blocks"
Texts contain single character detections, blocks are grouped detections
We use opencv to draw bounding boxes on text and block and then save it to path+_edited.png
Bounding boxes are provided by google vision api for both texts and blocks
"""
def detect_text(path):
    """Detects text in the file."""

    # Starts client with Google Cloud's Vision
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    cv2_img = cv2.imread(path)
    # cv2_img = cv2_img.resize((cv2_img.))
    window_name = 'image'

    response = client.text_detection(image=image)
    texts = response.text_annotations
    blocks = response.full_text_annotation.pages[0].blocks
    for block in blocks:
        raw_text = ""
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                raw_text += "".join([symbol.text for symbol in word.symbols])
        vertices = [
            (vertex.x, vertex.y) for vertex in block.bounding_box.vertices
        ]
        if len(raw_text.strip()) > 3:
            cv2_img = cv2.rectangle(cv2_img, vertices[0], vertices[2], (255, 200, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            x, y = vertices[0]
            org = (x, y + 10)

            # fontScale
            fontScale = .25

            # Blue color in BGR
            color = (0, 0, 255)

            # Line thickness of 2 px
            thickness = 1

            # Using cv2.putText() method
            print(f"Block: {raw_text}")
            cv2_img = cv2.putText(cv2_img, translate_with_deepl(raw_text), org, font, fontScale, color, thickness, cv2.LINE_AA)

    # print("Texts:")

    for text in texts:
        # print(f'\n"{text.description}"')

        vertices = [
            (vertex.x,vertex.y) for vertex in text.bounding_poly.vertices
        ]

        # print("bounds: {}".format(",".join(vertices)))
        # cv2_img = cv2.rectangle(cv2_img, vertices[0], vertices[2], (0,255,0), 2)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    cv2.imwrite(f"{path}_edited.png", cv2_img)

    return texts, blocks

"""
Make api call to openai's gpt, passing the text to translate
Returns response string ONLY
"""
def translate_with_gpt(text):
    messages = [{
        "role": "system",
        "content": "You are a translator. You translate any text to US-EN without losing native nuances, "
                   "even if you have to stray from a literal translation. "
    }, {
        "role": "user",
        "content": text
    }]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response["choices"][0]["message"]["content"]


def translate_with_llm(prompt):
    prompt_template = f'''A chat between a user and a translator that translates any language to English, faithfully, 
    and while preserving nuance, even at the cost of straying from a literal translation. USER: 呐朋君! ASSISTANT: Hey, 
    buddy! USER: {prompt} ASSISTANT: 

    '''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    return tokenizer.decode(output[0]).replace(prompt, "")


def translate_with_google(target: str, text: str) -> str:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    # print("Text: {}".format(result["input"]))
    # print("Translation: {}".format(result["translatedText"]))
    # print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result["translatedText"]


def translate_with_deepl(text):
    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": "DeepL-Auth-Key d6c76afd-77fc-1dc5-a789-aa0a234ae893:fx",
        "User-Agent": "MangaTranslator/1.0",
        "Content-Type": "application/json"
    }
    data = {
        "text": [text],
        "target_lang": "EN"
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response.json()["translations"][0]["text"]


def get_files(path):
    filenames = next(os.walk(path), (None, None, []))[2]
    return filenames


# init_openai()
# for panel in get_files(".\\sample_manga"):
#     blocks, texts = detect_text(f".\\sample_manga\\{panel}")
blocks, texts = detect_text(".\\samples\\img_4.png")

# print(get_files(".\\sample_manga"))