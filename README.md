# Manga Translator
 A translator for panel based materials in black and white.

I developed this for fun to prove that I could do it and the result was this, an application capable of accurately and quickly translating manga. It has some known kinks in its ability to distinguish between bubbles, but in practice, it should be able to cut down translation times significantly, while needing a bit more time for editing.

It uses a mix of computer vision, api calls, and LLMs to give translations. Google's Cloud Vision was used to locate text, and all located characters are then passed to translation services. Afterwards, the translated text is drawn onto the image and the image is reprocessed to clean up the original text and ensure the new text fits in the bubbles.
