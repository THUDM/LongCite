import os, json, re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import docx
import PyPDF2
from st_click_detector import click_detector

st.set_page_config(layout="wide")
st.title("LongCite Demo")
use_vllm = False # set True to use vllm for inference

MODEL_PATH = os.environ.get("MODEL_PATH", "THUDM/LongCite-glm4-9b")

@st.cache_resource
def load_model():
    model_path = MODEL_PATH
    if use_vllm:
        from vllm_inference import LongCiteModel
        model = LongCiteModel(
            model= model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=131072,
            gpu_memory_utilization=1,
        )
        tokenizer = model.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    return tokenizer, model

tokenizer, model = load_model()

def convert_to_txt(file):
    doc_type = file.name.split(".")[-1].strip()
    if doc_type in ["txt", "md", "py"]:
        data = [file.read().decode('utf-8')]   
    elif doc_type in ["pdf"]:
        pdf_reader = PyPDF2.PdfReader(file)
        data = [pdf_reader.pages[i].extract_text() for i in range(len(pdf_reader.pages))]  
    elif doc_type in ["docx"]:
        doc = docx.Document(file)
        data = [p.text for p in doc.paragraphs]
    else:
        st.error(f"ERROR: unsupported document type: {doc_type}")
    text = "\n\n".join(data)
    return text

def process_text(text):
    special_char={
        '&': '&amp;',
        '\'': '&apos;',
        '"': '&quot;',
        '<': '&lt;',
        '>': '&gt;',
        '\n': '<br>',
    }
    for x, y in special_char.items():
        text = text.replace(x, y)
    return text

html_styles = """<style>
    .reference {
        color: blue;
        text-decoration: underline;
    }
    .highlight {
        background-color: yellow;
    }
    .label {
        font-family: sans-serif;
        font-size: 16px;
        font-weight: bold;
    }
    .Bold {
        font-weight: bold;
    }
    .statement {
        background-color: lightgrey;
    }
</style>\n"""

def convert_to_html(statements, clicked=-1):
    html = html_styles + '<br><span class="label">Answer:</span><br>\n'
    all_cite_html = []
    clicked_cite_html = None
    idx = 0
    for i, js in enumerate(statements):
        statement, citations = process_text(js['statement']), js['citation']
        if clicked == i:
            html += f"""<span class="statement">{statement}</span>"""
        else:
            html += f"<span>{statement}</span>"
        if citations:
            cite_html = []
            idxs = []
            for c in citations:
                idx += 1
                idxs.append(str(idx))
                cite = '[Sentence: {}-{}\t|\tChar: {}-{}]<br>\n<span {}>{}</span>'.format(c['start_sentence_idx'], c['end_sentence_idx'], c['start_char_idx'], c['end_char_idx'],  'class="highlight"' if clicked==i else "", process_text(c['cite'].strip()))
                cite_html.append(f"""<span><span class="Bold">Snippet [{idx}]:</span><br>{cite}</span>""")
            all_cite_html.extend(cite_html)
            cite_num_html = """ <a href='#' class="reference" id={}>[{}]</a>""".format(i, ','.join(idxs))
            html += cite_num_html
        html += '\n'
        if clicked == i:
            clicked_cite_html = html_styles + """<br><span class="label">Citations of current statement:</span><br><div style="overflow-y: auto; padding: 20px; border: 0px dashed black; border-radius: 6px; background-color: #EFF2F6;">{}</div>""".format("<br><br>\n".join(cite_html))
    all_cite_html = html_styles + """<br><span class="label">All citations:</span><br>\n<div style="overflow-y: auto; padding: 20px; border: 0px dashed black; border-radius: 6px; background-color: #EFF2F6;">{}</div>""".format("<br><br>\n".join(all_cite_html).replace('<span class="highlight">', '<span>'))
    return html, all_cite_html, clicked_cite_html

@st.fragment
def render_answer(statements):
    answer_html, all_cite_html, clicked_cite_html = convert_to_html(statements, clicked=st.session_state.get("last_clicked", -1))
    col1, col2 = st.columns([4, 4])
    with col1:
        clicked = click_detector(answer_html)

    with col2:
        if clicked_cite_html:
            st.html(clicked_cite_html)
        st.html(all_cite_html)
    change = False
    if clicked != "":
        clicked = int(clicked)
        if "last_clicked" not in st.session_state:
            st.session_state["last_clicked"] = clicked
            change = True
        else:
            if clicked != st.session_state["last_clicked"]:
                st.session_state["last_clicked"] = clicked
                change = True
        if change:
            st.rerun(scope='fragment')

def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif', font_weight='normal'):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
        elem.style.fontWeight = '{font_weight}';
    </script>
    """
    st.components.v1.html(html)

col1, col2 = st.columns([4, 4])
context = None

with col1:
    uploaded_file = st.file_uploader("Upload a document (supported type: pdf, docx, txt, md, py)")

with col2:
    if uploaded_file is not None:
        context = convert_to_txt(uploaded_file)
        st.text_area("Document Content", context, height=270)

result = None
with col1:
    query = st.text_input("Question:")
    # change_label_style("Question:", font_size='16px', font_weight="bold")
    if st.button("Submit") and query:
        if context is None:
            st.error("Error: no uploaded document.")
        with st.spinner('running...'): 
            result = model.query_longcite(context, query, tokenizer=tokenizer, max_input_length=128000, max_new_tokens=1024)

if result:
    statements = result['all_statements']
    render_answer(statements)
