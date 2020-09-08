import crosslang_embed

phrase_list = ['hi', 'hello', 'bon nuit', 'bonjourno', 'boujour', 'bon soir', 'good night',
               'good evening']


mph = crosslang_embed.MultilangPhrase()

# translate and embed phrases
mph.process_phrases(phrase_list)

# Show languages and translations for all phrases ---

print(mph.dfmain)

# Find matches for any single phrase ---

print("find_matches_index:","salut")
df = mph.find_matches_index("salut", )
print( df[ df.dist<1.2].head(30) )


print("find_matches_index:","good evening")
df = mph.find_matches_index("good evening",)
print( df[ df.dist<1.2].head(10) )



# Plot phrases to see clustering ---

mph.umap_embeddings()

mph.plot_embeddings()

