import sys
import numpy as np
import os
import pandas as pd
from annoy import AnnoyIndex
import tensorflow as tf
print(tf.__version__)
print(sys.executable)

import tensorflow_text
import tensorflow_hub as hub
import os
import time
from google.cloud import translate


GOOGLE_CREDS_JSON = "path_to_google_creds_json"
GOOGLE_PROJECT_ID = "google_project_id"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=GOOGLE_CREDS_JSON


global usenc_hub_model
usenc_hub_model = None


def load_encoder_module():
  global usenc_hub_model
  if usenc_hub_model==None:

    module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
    # can replace this URL
    # 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
    # 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3']

    usenc_hub_model = hub.load(module_url)


def embed_text(input):
  """
  input: currently a pandas.Series but maybe list?
  """
  global usenc_hub_model
  return usenc_hub_model(input)





def sample_translate_texts(texts=["YOUR_TEXT_TO_TRANSLATE"], project_id=GOOGLE_PROJECT_ID):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = 'global'
    parent = "projects/{project_id}/locations/{location}".format(project_id=project_id,location=location)

    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        parent=parent,
        contents=texts,
        mime_type="text/plain",  # mime types: text/plain, text/html
        #source_language_code="en-US",
        target_language_code="en",
    )
    # Display the translation for each input text provided
    for translation in response.translations:
        print(u"Translated text: {}".format(translation.translated_text))
    return [translation for translation in response.translations]



def translate_all(text_list):
    """
    text_list : list of strings
    return: src_to_en_map, dataframe_rows
        src_to_en_map is the short version.
        dataframe_rows is the full version.

        src_to_en_map:  {'Pessoas': 'People', 'Negocio': 'Deal', ... }
        dataframe_rows: [{'en': 'People', 'src_lang': 'pt', 'src': 'Pessoas'}, ... ]
    """
    batch_len=100
    batch_start=0
    batch_end=batch_start+batch_len
    mapp={}
    rows=[]
    done=False
    while not done:
        batch=text_list[batch_start:batch_end]
        xlat=sample_translate_texts(batch)

        for idx,b in enumerate(batch):
            row={ 'en':xlat[idx].translated_text , 'src_lang':xlat[idx].detected_language_code, 'src':batch[idx] }
            #print(row)
            rows += [row]
            mapp[batch[idx]] = xlat[idx].translated_text

        batch_start+=batch_len
        batch_end=batch_start+batch_len

        if len(batch)<batch_len:
            done=True
    time.sleep(0.1)
    return mapp, rows



class MultilangPhrase():

    def __init__(self):
        load_encoder_module()


    def process_phrases( self, phrase_list):
        phrases=[q for q in phrase_list if type(q)==str]

        self.xlat_strlist_src = list(set(phrases))

        len(self.xlat_strlist_src)

        self.src_to_en_map, dataframe_rows = translate_all(self.xlat_strlist_src)
        self.src_to_en_map
        dataframe_rows[:2]

        dflang = pd.DataFrame(dataframe_rows)
        self.dfmain=dflang.set_index('src')
        self.dfmain

        dflang['en']

        # embed the *english* vector.  We dont have to do it that way.
        self.xlat_vector=embed_text(dflang['en'])
        self.xlat_vector
        self.xlat_srclang=dflang['src_lang']

        self.xlat_strlist_src = list(set(self.xlat_strlist_src))
        self.xlat_strlist_src[:3]


        embedding_dimensions = self.xlat_vector.shape[1]
        embedding_dimensions

        self.annoy_index = AnnoyIndex(embedding_dimensions, 'angular')  # Length of item vector that will be indexed

        for vec_idx, vec in enumerate(self.xlat_vector):
            self.annoy_index.add_item(vec_idx, vec.numpy())

        self.annoy_index.build(n_trees=10)


    def find_matches_index_raw( self, text ):
        """
        text: input string
        return: ( phrase_index_list, distance_list  )
        eg: ( [17, 18], [0.0, 0.6941506862640381,] )

        Need to have:
        t:  annoy index
        """

        # get query vector
        qvec = embed_text(text).numpy()[0]

        # lookup phrase indexes
        phrase_idxs,dists = self.annoy_index.get_nns_by_vector(qvec, 100,include_distances=True)

        # render chart
        pd.Series(dists).plot()

        return phrase_idxs,dists


    # ```
    # xlat_strlist_src ['Afuo yɛ ho ntotoeɛ',
    # xlat_srclang ['fr','ig'...
    # src_to_en_map {'Afuo yɛ ho ntotoeɛ': 'Let me hear you', ...
    # ```

    def find_matches_index( self, text, ):
        """
        text:  string to lookup
        xlat_strlist_src: list of all source strings
        xlat_srclang: list of srclangs for source strings
        annoy_index:
        """

        # lookup phrase indexes
        phrase_idxs,dists = self.find_matches_index_raw( text,  )

        rows=[]

        for idx,phrase_index in enumerate(phrase_idxs):
            #print(idx)
            row={'en':self.src_to_en_map[self.xlat_strlist_src[phrase_index]],
                 'phrase':self.xlat_strlist_src[phrase_index],
                 'dist':dists[idx],
                 'lang':self.xlat_srclang[idx]}
            rows += [row]

        df=pd.DataFrame(rows)
        return df.head(30)


    def umap_embeddings(self):
        import umap.umap_ as umap

        reducer = umap.UMAP(n_neighbors=3)
        self.umap_embedding = reducer.fit_transform(self.xlat_vector)
        self.umap_embedding.shape

        self.dfmain['x']=self.umap_embedding[:, 0]
        self.dfmain['y']=self.umap_embedding[:, 1]
        self.dfmain=self.dfmain.reset_index()
        print(self.dfmain)



    def plot_embeddings(self):

        import matplotlib.pyplot as plt
        import plotly.express as px

        plt.gca().set_aspect('equal', 'datalim')

        fig = px.scatter(self.dfmain, x='x', y='y', text='en', ) #,size_max=20size='pillar_counts_src' )
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=900, width=1000,
            title_text='Planets', font_size=9
        )
        fig.show()


        fig = px.scatter(self.dfmain, x='x', y='y', text='src',) #size='pillar_counts_src',size_max=100 )
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=1800, width=2400,
            title_text='Phrase embedding'
        )
        fig.show()


        plt.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], )
        plt.title('UMAP projection', fontsize=24);


