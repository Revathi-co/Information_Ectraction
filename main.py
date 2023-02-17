from tkinter import *
import urllib3
import docx  # pip install python-docx
import fitz  # pip install PyMuPDF
from bs4 import BeautifulSoup  # pip install bs4
from nltk.tokenize import word_tokenize  # pip install nltk
import re
import pandas as pd #pip install pandas
import numpy as np
import spacy  # pip install spacy
from tqdm import tqdm
import en_core_web_sm   # python -m spacy download en_core_web_sm

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import time


from sumy.parsers.plaintext import PlaintextParser    # pip install sumy
from sumy.nlp.tokenizers import Tokenizer
# Import the LexRank summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration     # pip install torch, transformers, pip install sentencepiece


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from nltk import PorterStemmer


# load english language model
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])
nlp1 = en_core_web_sm.load()
nlp2 = spacy.load('en_core_web_lg')


class MainWindow(Frame):
    """ Class to handle window"""

    def __init__(self, master=None):
        Frame.__init__(self, master, bg="#006699")
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Text Summarization GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        self.words = []

        # creating a buttons, entry, textbox instance
        self.label = Label(self, text="Enter url:", relief=RAISED)
        self.entry_url = Entry(width=50)
        self.entry_url.insert(0, "")
        self.get_url = Button(self, text="Get URL", command=self.client_get_url)
        self.label2 = Label(self, text="Enter File path:", relief=RAISED)
        self.file_in = Entry(width=50)
        self.get_file = Button(self, text="Get File", command=self.client_get_file)
        self.text_pro_button = Button(self, text="Text processing", command=self.openTextProWindow)
        self.save_button = Button(self, text="Save", command=self.client_save)
        self.quitButton = Button(self, text="Exit", command=self.client_exit)
        self.t = Text(self, width=60, height=15)

        self.frame = Frame(root, bg="red")
        self.frame.pack()
        # placing the button on my window
        self.quitButton.place(x=430, y=350)
        self.get_url.place(x=410, y=20)
        self.label.place(x=5, y=20)
        self.entry_url.place(x=100, y=20)
        self.label2.place(x=5, y=60)
        self.file_in.place(x=100, y=60)
        self.get_file.place(x=410, y=60)
        self.text_pro_button.place(x=205, y=350)
        self.save_button.place(x=5, y=350)
        self.t.place(x=5, y=100)

    def client_exit(self):
        """Exit function"""
        exit()

    def client_save(self):
        file = open("file.txt", "w+")
        file.write(" Text in "+self.entry_url+":")
        for w in self.words:
            file.write(w)
        file.close()

    def client_get_file(self):
        self.t.delete(1.0, END)
        path = self.file_in.get()

        if path.endswith(".pdf"):
            global st
            st= time.time()
            with fitz.open(path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            self.t.insert("end", text)
            self.words.append(text)

        elif path.endswith(".docx"):
            st = time.time()
            doc = docx.Document(path)
            for para in doc.paragraphs:
                self.words.append(para.text)
            self.t.insert("end", '\n'.join(self.words))

        else:
            file = open(path, "r+")
            st = time.time()
            line = file.readlines()
            self.t.insert("end", line)
            for l in line:
                self.words.append(l)

    def client_get_url(self):
        self.t.delete(1.0, END)
        """ Clean text operations """
        global st
        st = time.time()
        url = self.entry_url.get()
        http = urllib3.PoolManager()
        page = http.request('GET', url)
        soup = BeautifulSoup(page.data, "html.parser")
        self.t.delete(1.0, END)
        for pp in soup.find_all('p'):
            self.t.insert("end", pp.text)
            self.words.append(pp.text)

    def openTextProWindow(self):

        newTextProWindow = TextProWindow(self.words, root)


class TextProWindow(Toplevel):

    def __init__(self, text, master=None):
        super().__init__(master=master, bg="#006699")
        self.title("Text Processing")
        self.text = text
        self.geometry("500x400")
        label = Label(self, text="Text Processing")
        label.pack(side=TOP, pady=15)
        self.init_text_pro_window()

    def init_text_pro_window(self):
        self.textarea = Text(self, width=60, height=12)
        self.client_text_processor()

        self.var = IntVar()
        self.R1 = Radiobutton(self, width=64, text="Text Summarization", variable=self.var, value=1,
                              command=self.sel)
        self.R1.pack(anchor=W, pady=2)

        self.R2 = Radiobutton(self, width=64, text="Information Extraction", variable=self.var, value=2,
                              command=self.sel)
        self.R2.pack(anchor=W, pady=2)

        self.R3 = Radiobutton(self, width=64, text="Question Answering", variable=self.var, value=3,
                              command=self.sel)
        self.R3.pack(anchor=W, pady=2)
        self.frame = Frame(root, bg="red")
        self.frame.pack()
        self.textarea.place(x=5, y=40)
        self.R1.place(x=8, y=260)
        self.R2.place(x=8, y=300)
        self.R3.place(x=8, y=340)

    def clean(self,text):
        # removing paragraph numbers
        text = re.sub('[0-9]+.\t', '', str(text))
        # removing new line characters
        text = re.sub('\n ', '', str(text))
        text = re.sub(',\n ', ',', str(text))
        text = re.sub('\n', ' ', str(text))
        text = re.sub("\n'", ' ', str(text))
        text = re.sub('\n\",', ' ', str(text))
        text = re.sub('\n,', ',', str(text))
        # removing apostrophes
        text = re.sub("'s", '', str(text))
        # removing hyphens
        text = re.sub("-", ' ', str(text))
        text = re.sub("— ", '', str(text))
        # removing quotation marks
        text = re.sub('\"', '', str(text))
        # removing any reference to outside text
        text = re.sub("[\(\[].*?[\)\]]", "", str(text))

        # split sentences and questions
        text = re.split('[.?]', text)

        return text

    def client_text_processor(self):
        # self.text = self.clean(self.text)
        self.textarea.delete(1.0, END)

        self.textarea.insert("end", "No. of sentences: " + str(len(self.text)))
        stemmer = PorterStemmer()
        word = []
        self.textarea.insert("end", " \nNo.of Normal words : ")
        for sent in self.text:
            word.extend(word_tokenize(sent))
        self.textarea.insert("end", len(word))
        stemmed_words = [stemmer.stem(i) for i in word]
        self.textarea.insert("end", " \nNo. of Stemmed words : " + str(len(stemmed_words)))
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(" ".join(self.text))
        tokens = [token for token in tokens if token not in stopwords.words('english') and not str.isdigit(token)]
        self.textarea.insert("end", len(tokens))
        self.textarea.insert("end", "\n\n 10 Most Frequently used words : " + str(Counter(tokens).most_common(10)))
        et = time.time()
        elapsed_time = et - st
        print('text processing time:', elapsed_time, 'seconds')


    def sel(self):
        option = self.var.get()
        if option == 1:
            newTextSumWindow = TextSumWindow(self.text, self.master)

        elif option == 2:
            newInfoExWindow = InfoExtractWindow(self.text, self.master)
        else:
            newQuesAnsWindow = QuesAnsWindow(self.text, self.master)


class TextSumWindow(Toplevel):

    def __init__(self, text, master=None):
        super().__init__(master=master, bg="#006699")
        self.title("Text Summarization")
        self.text = text
        self.geometry("500x420")
        label = Label(self, text="Text Summarization",width=60, height=2)
        label.pack(side=TOP, pady=15)
        self.init_text_pro_window()

    def init_text_pro_window(self):
        self.textarea = Text(self, width=60, height=10)

        self.var = IntVar()
        self.R1 = Radiobutton(self, width=30, text="Extractive Text Summarization", variable=self.var, value=1,
                              command=self.sel)
        self.R1.pack(anchor=W, pady=2)

        self.R2 = Radiobutton(self, width=30, text="Abstractive Text Summarization", variable=self.var, value=2,
                              command=self.sel)
        self.R2.pack(anchor=W, pady=2)

        self.R3 = Radiobutton(self, width=30, text="Both", variable=self.var, value=3,
                              command=self.sel)
        self.R3.pack(anchor=W, pady=2)
        self.frame = Frame(root, bg="red")
        self.frame.pack()
        self.textarea.place(x=5, y=140)
        self.R1.place(x=8, y=65)
        self.R2.place(x=250, y=65)
        self.R3.place(x=150, y=100)

    def sel(self):
        option = self.var.get()
        if option == 1:
            st = time.time()
            self.extractive_sum(self.text)
            et = time.time()
            elapsed_time = et - st
            print('extractive Execution time:', elapsed_time, 'seconds')
        elif option == 2:
            st = time.time()
            self.abstractive_sum(self.text,0)
            et = time.time()
            elapsed_time = et - st
            print('abs extractive Execution time:', elapsed_time, 'seconds')
        else:
            st = time.time()
            self.extractive_sum(self.text)
            self.abstractive_sum(self.text,1)
            et = time.time()
            elapsed_time = et - st
            print('both extractive Execution time:', elapsed_time, 'seconds')

    def extractive_sum(self,text):
        # Initializing the parser
        preprocess_text = ",".join(text)
        my_parser = PlaintextParser.from_string(preprocess_text, Tokenizer('english'))
        # Creating a summary of 3 sentences.
        lex_rank_summarizer = LexRankSummarizer()
        lexrank_summary = lex_rank_summarizer(my_parser.document, sentences_count=5)

        self.textarea.delete(1.0, END)
        self.textarea.insert("end", "Extractive Summarized text: \n")
        # Printing the summary
        for sentence in lexrank_summary:
            self.textarea.insert("end", sentence)


    def abstractive_sum(self, text, flag):
        # Instantiating the model and tokenizer
        my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        summary =[]
        pre_text = ",".join(text).strip().replace("\n", "")
        val_list = []
        # Concatenating the word "summarize:" to raw text
        if len(pre_text) > 512:
            size= len(pre_text)
            value= size//512
            for i in range(1,value+1):
                val_list.append(i*512)

        for i in range(len(val_list)):
            if val_list[i] ==512:
                text = pre_text[:val_list[i]]
            else:
                text = pre_text[val_list[i-1]:val_list[i]]

            sum_text = "summarize:" + text

            # encoding the input text
            input_ids = tokenizer.encode(sum_text, return_tensors='pt', max_length=512)

            # Generating summary ids
            summary_ids = my_model.generate(input_ids)

            # Decoding the tensor and printing the summary.
            t5_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary.append(t5_summary)

        if flag==0:
            self.textarea.delete(1.0, END)
            self.textarea.insert("end", "Abstractive Summarized text: \n"+",".join(summary))

        else:
            self.textarea1 = Text(self, width=60, height=5)
            self.textarea1.place(x=5, y=320)
            self.textarea1.delete(1.0, END)
            self.textarea1.insert("end", "Abstractive Summarized text: \n" + ",".join(summary))




class InfoExtractWindow(Toplevel):

    def __init__(self, text, master=None):
        super().__init__(master=master, bg="#006699")
        self.title("Information Extraction")
        self.text = text
        self.geometry("950x700")
        label = Label(self, text="Information Extraction",width=60, height=2)
        label.pack(side=TOP, pady=15)
        self.init_info_ext_window()


    def init_info_ext_window(self):
        self.textarea = Text(self, width=115, height=30)

        self.var = IntVar()
        self.R1 = Radiobutton(self, width=60, text="Information Extraction", variable=self.var, value=1,
                              command=self.sel)
        self.R1.pack(anchor=W, pady=2)

        self.R2 = Radiobutton(self, width=60, text="Name Entity Recognition", variable=self.var, value=2,
                              command=self.sel)
        self.R2.pack(anchor=W, pady=2)

        self.R3 = Radiobutton(self, width=60, text="Event Extraction", variable=self.var, value=3,
                              command=self.sel)
        self.R3.pack(anchor=W, pady=2)
        self.frame = Frame(root, bg="red")
        self.frame.pack()
        self.textarea.place(x=10, y=180)
        self.R1.place(x=10, y=85)
        self.R2.place(x=472, y=85)
        self.R3.place(x=250, y=120)

    def sel(self):
        row_list = []
        option = self.var.get()
        if option == 1:
            # create a df containing sentence and its output for rule 3
            st = time.time()
            row_list = []

            # df2 contains all the sentences from all the speeches
            for i in range(len(self.text)):
                sent = self.text[i]
                output = self.rule3_mod(sent)
                dict1 = {'Sent': sent, 'Output': output}
                row_list.append(dict1)
            self.textarea.delete(1.0, END)
            df_rule3_mod = pd.DataFrame(row_list)
            for i in range(len(df_rule3_mod)):
                self.textarea.insert("end", "\n[ Paragragh "+str(i)+"]  " + str(df_rule3_mod.loc[i, 'Sent']) + "\n")
                self.textarea.insert("end"," Information Extracted => "+ str(df_rule3_mod.loc[i,'Output'])+"\n")
                self.textarea.insert("end", ("-" * 110))
            et = time.time()
            elapsed_time = et - st
            print('Infomation extraction Execution time:', elapsed_time, 'seconds')

        elif option == 2:
            st = time.time()
            text = " ".join(self.text)
            doc = nlp1(text)
            for X in doc.ents:
                text= X.text
                label = X.label_
                dict = {'Text': text, 'Label': label}
                row_list.append(dict)
            df_ner = pd.DataFrame(row_list)

            self.textarea.delete(1.0, END)
            label_list = set(df_ner["Label"])
            for label in label_list:
                self.textarea.insert("end", "\n-> "+ label+("\t"*5)+str(df_ner[df_ner["Label"]== label]["Text"].unique())+"\n")
                self.textarea.insert("end",("-"*100))

            et = time.time()
            elapsed_time = et - st
            print('NER Execution time:', elapsed_time, 'seconds')

        else:
            st = time.time()
            self.textarea.delete(1.0, END)
            result = self.get_central_vector(self.text)
            self.textarea.insert("end",str(result))
            et = time.time()
            elapsed_time = et - st
            print('Event Extraction Execution time:', elapsed_time, 'seconds')


    # rule 0
    def rule0(self, text, index):

        doc = nlp(text)
        token = doc[index]
        entity = ''

        for sub_tok in token.children:
            if (sub_tok.dep_ in ['compound', 'amod']):  # and (sub_tok.pos_ in ['NOUN','PROPN']):
                entity += sub_tok.text + ' '

        entity += token.text

        return entity

    # rule 3 function
    def rule3_mod(self, text):

        doc = nlp(text)
        sent = []
        for token in doc:
            if token.pos_ == 'ADP':
                phrase = ''
                if token.head.pos_ == 'NOUN':

                    # appended rule
                    append = self.rule0(text, token.head.i)
                    if len(append) != 0:
                        phrase += append
                    else:
                        phrase += token.head.text
                    phrase += ' ' + token.text

                    for right_tok in token.rights:
                        if (right_tok.pos_ in ['NOUN', 'PROPN']):
                            right_phrase = ''
                            # appended rule
                            append = self.rule0(text, right_tok.i)
                            if len(append) != 0:
                                right_phrase += ' ' + append
                            else:
                                right_phrase += ' ' + right_tok.text

                            phrase += right_phrase

                    if len(phrase) > 2:
                        sent.append(phrase)
        return sent


    def get_mean_vectors(self,sents):
        a=np.zeros(300)
        for sent in sents:
            a=a+nlp2(sent).vector
        return  a/len(sents)

    def get_central_vector(self,sents):
        from sklearn.metrics import pairwise_distances_argmin_min
        vecs=[]
        for sent in sents:
            doc = nlp2(sent)
            vecs.append(doc.vector)
        mean_vec = self.get_mean_vectors(sents)
        index= pairwise_distances_argmin_min(np.array([mean_vec]),vecs)[0][0]
        return sents[index]


class QuesAnsWindow(Toplevel):
    def __init__(self, text, master=None):
        super().__init__(master=master, bg="#006699")
        self.title("Question Answering System")
        self.text = text
        self.geometry("500x420")
        label = Label(self, text="Question Answering System",width=60, height=2)
        label.pack(side=TOP, pady=15)
        self.init_ques_ans_window()

    def init_ques_ans_window(self):
        label1 = Label(self, text="Enter Question:", width=30, height=1)
        self.textarea = Text(self, width=60, height=10)
        self.question_text = Entry(self, width= 75)
        self.get_ans = Button(self, text="Get Answer", command=self.get_answer)
        self.frame = Frame(root, bg="red")
        self.frame.pack()
        self.textarea.place(x=5, y=180)
        self.question_text.place(x=5, y=90)
        label1.place(x=5,y=70)
        self.get_ans.place(x=320,y= 140)

    def get_answer(self):
        # Model
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question = self.question_text.get()
        paragraph = " ".join(self.text)

        encoding = tokenizer.encode_plus(text=question, text_pair=paragraph[:512])

        inputs = encoding['input_ids']  # Token embeddings
        sentence_embedding = encoding['token_type_ids']  # Segment embeddings
        tokens = tokenizer.convert_ids_to_tokens(inputs)  # input tokens

        start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]), return_dict=False)

        start_index = torch.argmax(start_scores)

        end_index = torch.argmax(end_scores)

        answer = ' '.join(tokens[start_index:end_index + 1])
        self.textarea.delete(1.0, END)
        self.textarea.insert("end", answer)

root = Tk()

# size of the window
root.geometry("500x400")
app = MainWindow(root)
root.mainloop()

#HELLEN
# extractive Execution time: 0.14369440078735352 seconds
# abs extractive Execution time: 19.038907766342163 seconds


