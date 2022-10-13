import re
from string import punctuation

import benepar
import spacy
from nltk import tokenize
from nltk.tree import Tree
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from summa.summarizer import summarize
from termcolor import colored
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

print(colored('Loading tf models ...', 'cyan'))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gtp2_model = TFGPT2LMHeadModel.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id)
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_sm")
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

print(colored('Done: TF models loaded ...', 'cyan'))


# Remove sentences containing punctuations, double quotes, and single quotes
def remove_unwanted_sentences(sentences):
    output = []
    for sent in sentences:
        single_quotes_present = len(re.findall(r"['][\w\s.:;,!?\\-]+[']",
                                               sent)) > 0
        double_quotes_present = len(re.findall(r'["][\w\s.:;,!?\\-]+["]',
                                               sent)) > 0
        question_present = "?" in sent
        if single_quotes_present or double_quotes_present or question_present:
            continue
        else:
            output.append(sent.strip(punctuation))
    return output


def get_important_sentences(text, ratio=0.3):
    sentences = summarize(text, ratio=ratio)
    sentences_list = tokenize.sent_tokenize(sentences)
    sentences_list = [re.split(r'[:;]+', x)[0] for x in sentences_list]

    # Remove very short sentences less than 30 characters and long sentences greater than 150 characters
    filtered_list_short_sentences = [
        sent for sent in sentences_list if len(sent) > 30 and len(sent) < 150
    ]
    return filtered_list_short_sentences


def chop_off_sentence(sentence) -> list[str, str]:
    sentence = sentence.rstrip('?:!.,;').split(".")[0]
    tree = construct_constituency_tree(sentence)
    last_nounphrase, last_verbphrase = get_right_most_VP_or_NP(tree)
    last_nounphrase_flattened = get_flattened(last_nounphrase)
    last_verbphrase_flattened = get_flattened(last_verbphrase)

    if len(last_nounphrase_flattened) < 0.6 * len(sentence) and len(
            last_verbphrase_flattened) < 0.6 * len(sentence):
        chopped = max(last_nounphrase_flattened,
                      last_verbphrase_flattened,
                      key=len)
    else:
        chopped = min(last_nounphrase_flattened,
                      last_verbphrase_flattened,
                      key=len)
    chopped = chopped.replace(" - ", "-").replace("  ", " ")
    partial_sentence = sentence.replace(chopped, "")
    return [partial_sentence, chopped]


def complete_phrase(phrase, full_sentence):
    num_return_sequences = 5
    encoded_input = tokenizer.encode(phrase, return_tensors='tf')
    max_length = len(phrase.split()) + max(40, 0.5 * len(phrase.split()))
    outputs = gtp2_model.generate(
        encoded_input,
        max_length=max_length,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        top_p=0.80,
        top_k=60,
        repetition_penalty=10.0,
    )
    sentences = []
    for i in range(num_return_sequences):
        res = tokenizer.decode(outputs[i], skip_special_tokens=True).replace(
            u'\xa0', u' ')
        res = res.split(".")
        if len(res) > 1:
            sentences.append(res[0].replace("  ", " "))

    # Compare to the original phrase and return the dissimilar phrase
    if full_sentence:
        sentence_embeddings = bert_model.encode([full_sentence] + sentences)
        if len(sentence_embeddings) < 2: return []
        similarities = cosine_similarity([sentence_embeddings[0]],
                                         sentence_embeddings[1:])
    return sorted(sentences, key=lambda x: similarities[0][sentences.index(x)])


def construct_constituency_tree(sentence):
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    tree_string = sent._.parse_string
    tree = Tree.fromstring(tree_string)
    return tree


def get_flattened(t):
    sent_str_final = ""
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_str_final = [" ".join(sent_str)]
        sent_str_final = sent_str_final[0]
    return sent_str_final


def get_right_most_VP_or_NP(parse_tree, last_NP=None, last_VP=None):
    if len(parse_tree.leaves()) == 1:
        return last_NP, last_VP
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    return get_right_most_VP_or_NP(last_subtree, last_NP, last_VP)


def generate_tf_questions(text):
    questions = []
    sentences = get_important_sentences(text)
    for sentence in sentences:
        questions.append((sentence, "True", ["True", "False"]))
        phrase, end = chop_off_sentence(sentence)
        res = complete_phrase(phrase, sentence)
        if res:
            index = len(res) // 2
            questions.append((res[index], "False", ["True", "False"]))
    return questions
