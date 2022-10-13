import numpy as np
from nltk.corpus import wordnet
from termcolor import colored
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer,
                          TokenClassificationPipeline)
from transformers.pipelines import AggregationStrategy


# Extract the key sentences from the text
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs)

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])


def get_key_sentences(text):
    text = text.replace("\n", " ")
    keyphrases = extractor_model(text)
    print(colored(f"Keyphrases: {str(keyphrases)}", "green"), )

    result = []
    for sentence in text.split("."):
        for keyphrase in keyphrases:
            if keyphrase in sentence:
                result.append([sentence, keyphrase])
    return result


def get_question(sentence, answer):
    text = "context: {} answer: {} </s>".format(sentence, answer)
    max_len = 256
    encoding = question_tokenizer.encode_plus(text,
                                              max_length=max_len,
                                              pad_to_max_length=True,
                                              return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding[
        "attention_mask"]

    outs = question_model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   early_stopping=True,
                                   num_beams=5,
                                   num_return_sequences=1,
                                   no_repeat_ngram_size=2,
                                   max_length=200)

    dec = [question_tokenizer.decode(ids) for ids in outs]
    question = dec[0].replace("question:",
                              "").replace("</s>",
                                          "").strip().replace("<pad>", "")
    question = question.strip()
    return question


def get_distractors_wordnet(syn, word):
    distractors = []
    word = word.lower()
    orig_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name and name not in distractors:
            distractors.append(name)
    return distractors[:3]


def generate_mcqs(text):
    result = []
    sentences = get_key_sentences(text)
    for context, answer in sentences:
        question = get_question(context, answer)
        #  Generate options for this question
        options = [answer]
        synset_to_use = wordnet.synsets(answer, 'n')
        if (len(synset_to_use) > 1):
            print(colored(f"Multiple synsets found for {answer}", "red"))

        if synset_to_use:
            synset_to_use = synset_to_use[0]
            options += get_distractors_wordnet(synset_to_use, answer)
        result.append((question, answer, options))

    return result


# Load the models
print(colored("Loading mcq models...", "cyan"))

question_model = T5ForConditionalGeneration.from_pretrained(
    'ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')
extractor_model = KeyphraseExtractionPipeline(
    model="ml6team/keyphrase-extraction-kbir-inspec")
