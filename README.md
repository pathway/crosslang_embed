# crosslang_embed

Process multilingual phrases.
Combines translation, embedding search, and embedding visualization.

Partly based on https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb

### Install

Prerequisites:
```
pip install -r requirements.txt
```

### Usage

Setup your google cloud platform credentials:
```
GOOGLE_CREDS_JSON = "path_to_google_creds_json"
GOOGLE_PROJECT_ID = "google_project_id"
```

Enter or load list of phrases you wish to translate:
```
phrase_list = ['hi', 'hello', 'bon nuit', 'bonjourno', 'boujour', 'bon soir', 'good night',
               'good evening']
```

Process phrases to translate them, get embeddings, and load embeddings.
```
mph.process_phrases(phrase_list)
```

Show all translations:
```
print(mph.dfmain)
```

Find matches for a single phrase:
```
df = mph.find_matches_index("salut", )
print( df[ df.dist<1.2].head(30) )
```

Cluster phrases:
```
mph.umap_embeddings()
mph.plot_embeddings()
```
