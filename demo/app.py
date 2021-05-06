import streamlit as st
import torch
import transformers
import time


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def inference(input, task="correction", model="Tiny T5", num_beams=None,
              early_stopping=None, no_repeat_ngram_size=None, top_k=None):
    input = tokenizer.encode_plus(f"{task}: "+input, return_tensors="pt",
                                                     truncation=True, padding='max_length')
    out = models[model].generate(input_ids=input["input_ids"],
                     attention_mask=input["attention_mask"],
                     max_length=256, num_beams=num_beams, early_stopping=early_stopping,
                                 no_repeat_ngram_size=no_repeat_ngram_size, top_k=top_k,
                                 do_sample=True if top_k is not None else False
                                    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def parsing_(sent):
    doc = nlp(sent)
    docsoup = []

    for token in doc:
        if token.is_punct:
            docsoup.append(str(token))
        else:
            info = [str(token.dep_)]
            info += [str(token.pos_)]
            info += [key[key.find("_") + 1:] for key in nlp.vocab.morphology.tag_map[token.tag_].keys() if
                     type(key) == str]
            docsoup.append("_".join(info))

    parsing = " ".join(docsoup)
    ret = "sentence: " + sent + " parsing: " + parsing
    return ret


@st.cache(allow_output_mutation=True)
def load_model():
    path = "./models/T5-small_distilled/"
    tokenizer = transformers.T5TokenizerFast.from_pretrained(path)
    tiny_model = transformers.T5ForConditionalGeneration.from_pretrained(path)
    tiny_model.eval()

    path = "./models/T5-base/"
    base_model = transformers.T5ForConditionalGeneration.from_pretrained(path)
    base_model.eval()

    path = "./models/T5-base-quantized/"
    model_int8 = torch.quantization.quantize_dynamic(
        base_model.to("cpu"),  # the original model
        {torch.nn.Linear: torch.quantization.default_dynamic_qconfig},  # a set of layers to dynamically quantize
        dtype=torch.qint8)
    model_int8.load_state_dict(torch.load(path+"model.checkpoint"))
    model_int8.eval()

    tiny_model_int8 = torch.quantization.quantize_dynamic(
        tiny_model.to("cpu"),  # the original model
        {torch.nn.Linear: torch.quantization.default_dynamic_qconfig},  # a set of layers to dynamically quantize
        dtype=torch.qint8)
    tiny_model_int8.eval()

    models = {"Tiny T5": tiny_model,
              "T5-base": base_model,
              "T5-base-quantized": model_int8,
              "Tiny T5 quantized": tiny_model_int8
              }
    return models, tokenizer


def main():
    st.header("Neural Grammar error correction")
    st.text("This is demo Grammar error correction models based on T5")

    st.sidebar.title("Menu")
    generation = st.sidebar.selectbox("Generation methods(that`s can improve quality)", ("Greedy Search", "Beam search", "Top-K Sampling"))
    num_beams = early_stopping = no_repeat_ngram_size = top_k = None
    if generation == "Beam search":
        num_beams = st.sidebar.slider('Num beams', 0, 1, 10)
        early_stopping = st.sidebar.checkbox('Early stopping')
        no_repeat_ngram_size = st.sidebar.slider('No repeat ngram size', 0, 1, 10)
    elif generation == "Top-K Sampling":
        top_k = st.sidebar.slider('Top-K', 0, 1, 100)

    dataset = st.selectbox("Dataset", ("correction", "bea", "conll", "jfleg"))
    model = st.selectbox("Model", ("T5-base", "T5-base-quantized", "Tiny T5", "Tiny T5 quantized"))

    text = st.text_area(value="i very very like nerual networks and google .",
                        label="Text")

    if len(text):
        start_time = time.time()
        if dataset == "bea" or dataset == "conll":
            text = "sentence: " + text

        out = inference(text, task=dataset, model=model, num_beams=num_beams,
                        early_stopping=early_stopping, no_repeat_ngram_size=no_repeat_ngram_size, top_k=top_k).split(" ")
        end_time = time.time()
        text = text.split(" ")
        t = "<div>"

        for item in out:
            if item not in text:
                t += f" <span class='highlight blue'>{item}</span>"

            else:
                t += " " + item

        t += " </div>"
        st.markdown(t, unsafe_allow_html=True)

        st.text(f"Inference time {end_time - start_time} s")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    local_css("style.css")
    models, tokenizer = load_model()

    #nlp = spacy.load("en")

    main()