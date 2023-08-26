import pandas as pd

# Load the df
df_main = pd.read_csv('lingbuzz_2_7448.csv')

# Drop nan entries
df_main.dropna(subset=['Title'], inplace=True)

# Date format to the Date column
df_main['Date'] = pd.to_datetime(df_main['Date'])
df_main['Date'] = df_main['Date'].dt.strftime('%Y-%m')

#%%

def keyword_checker(cell):
    target_keywords = [ 'phonology', 'morphology', 'syntax', 'semantics']
    contained_keywords = []
    for word in target_keywords:
        if word in cell:
            contained_keywords.append(word)
    replacement = ' + '.join(contained_keywords)
    return replacement

df_key = pd.DataFrame()

df_key['Id'] = df_main['Id']
df_key['Year'] = pd.to_datetime(df_main['Date']).dt.year
df_key['Year'] = df_key['Year'].apply(str)
df_key['Disciplines'] = df_main['Keywords'].apply(keyword_checker)

keyword_counts = df_key.groupby(['Year', 'Disciplines']).size().reset_index(name='count')

combinations_to_drop = ['', 'morphology + semantics', 'phonology + morphology + semantics', 'phonology + morphology + syntax', 'phonology + morphology + syntax + semantics', 'phonology + semantics', 'phonology + syntax', 'phonology + syntax + semantics']
keyword_counts = keyword_counts.drop(keyword_counts[keyword_counts['Disciplines'].isin(combinations_to_drop)].index)

#%%
import matplotlib.pyplot as plt

# pivoting
pivot_df = keyword_counts.pivot_table(index='Year', columns='Disciplines', values='count', fill_value=0)
pivot_df = pivot_df.drop(pivot_df.index[:16])
pivot_df = pivot_df.drop(pivot_df.index[-1])

pivot_df = pivot_df[['syntax', 'semantics', 'syntax + semantics',
                     'morphology + syntax', 'phonology', 'morphology + syntax + semantics',
                     'morphology', 'phonology + morphology']]

# Plot the histogram
pivot_df.plot(kind='line', stacked=False, figsize=(16, 9), linewidth=3)

# Add labels and title
plt.xlabel('')
#plt.ylabel('Number of manuscripts')
plt.title('Keyword trends in Lingbuzz (2002-2022)', fontsize=18)
plt.legend(fontsize=12)

#%%

from collections import Counter

def key_counter(n = None):
    '''
    Parameters
    ----------
    n : int, optional
        Takes a number n and returns the n most common keywords. If no n is introduced, it returns a dictionary with counts for all keywords.

    Returns
    -------
    dictionary
        A dictionary with counts of keywords 

    '''
    keywords = []
    for cell in df_main['Keywords']:
        keywords += cell.split(', ')
    keywords = [key.lower() for key in keywords]
    if n == None:  
        return dict(Counter(keywords))
    else:
        return dict(Counter(keywords).most_common(n))

        
keywords = key_counter(150)      
        
#%%

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800, background_color='white')

wordcloud.generate_from_frequencies(keywords)
        
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
        
#%%        

def author_counter(n = None):
    '''
    Parameters
    ----------
    n : int, optional
        Takes a number n and returns the n most frquent authors. If no n is introduced, it returns a dictionary with counts for all authors.

    Returns
    -------
    dictionary
        A dictionary with counts of keywords 

    '''
    authors = []
    for cell in df_main['Authors']:
        authors += cell.split(', ')
    if n == None:  
        return dict(Counter(authors))
    else:
        return dict(Counter(authors).most_common(n))

        
most_manuscripts_authors = author_counter(10)

least_manuscripts_authors = author_counter()
        
#%%

def authors_downloads():
    authors_down = {}
    for i in range(len(df_main)):
        aut_paper = df_main.iloc[i, 2].split(', ')
        downloads = df_main.iloc[i, 6]
        for person in aut_paper:
            if person in authors_down.keys():
                authors_down[person] += downloads
            else:
                authors_down[person] = downloads
    return authors_down

down_authors = authors_downloads()
        
#%% 

def network_connections():
    connections = {}
    for cell in df_main['Authors']:
        authors = cell.split(', ')
        conn_by_paper = []
        for a in authors:
            for b in authors:
                if a != b:
                    conn = frozenset({a, b})
                    conn_by_paper.append(conn)
        conn_by_paper = set(conn_by_paper)
        for i in conn_by_paper:
            if i in connections.keys():
                connections[i] += 1
            else:
                connections[i] = 1
    return {tuple(key): value for key, value in connections.items()}

connections = network_connections()
        
        
#%%      
import networkx as nx

G = nx.Graph()

for key, value in connections.items():
    G.add_edge(key[0], key[1], weight=value)
     
        
is_connected = nx.is_connected(G)       
        
# Find all connected components
connected_components = list(nx.connected_components(G))

# Find the largest connected component
main_connected_chunk = max(connected_components, key=len)

# Create a subgraph with the main connected chunk
subgraph = G.subgraph(main_connected_chunk)

plt.figure(figsize=(216, 144))

# Plot the main connected chunk
pos = nx.spring_layout(G, k=0.2)  # Choose a layout for visualization
nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=200, font_size=15, font_color='red', font_weight='bold', edge_color='darkgray')

plt.show();      
        
#%%

collaborators = sorted(list(G.degree()), key=lambda x: x[1], reverse=True)

# closer 
closeness_centrality = nx.closeness_centrality(G)

# Calculate betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)

# Find nodes with higher closeness centrality
higher_closeness_nodes = {node for node, closeness in closeness_centrality.items() if closeness == max(closeness_centrality.values())}

# Find nodes with higher betweenness centrality
higher_betweenness_nodes = {node for node, betweenness in betweenness_centrality.items() if betweenness == max(betweenness_centrality.values())}


#%%

# per language # this is not right. separation is required

languages = []

# This is a custom list of reference languages

target_keywords = [
    'Brazilian Portuguese', 'Canadian French', 'English', 'French Creole', 'Haitian Creole', 'Navajo',
    'Quechua', 'Spanish', 'Catalan', 'Danish', 'Dutch', 'Faroese', 'Finnish', 'Flemish', 'French', 'German',
    'Greek', 'Icelandic', 'Italian', 'Norwegian', 'Portuguese', 'Spanish', 'Swedish', 'UK English / British English',
    'Belarusian', 'Bosnian', 'Bulgarian', 'Croatian', 'Czech', 'Estonian', 'Hungarian', 'Latvian', 'Lithuanian',
    'Macedonian', 'Polish', 'Romanian', 'Russian', 'Serbian', 'Slovak', 'Slovenian', 'Turkish', 'Ukrainian',
    'Amharic', 'Dinka', 'Ibo', 'Kirundi', 'Mandinka', 'Nuer', 'Oromo', 'Kinyarwanda', 'Shona', 'Somali', 'Swahili',
    'Tigrigna', 'Wolof', 'Xhosa', 'Yoruba', 'Zulu', 'Arabic', 'Farsi', 'Hebrew', 'Kurdish', 'Pashtu', 'Punjabi',
    'Urdu', 'Armenian', 'Azerbaijani', 'Georgian', 'Kazakh', 'Mongolian', 'Turkmen', 'Uzbek', 'Bengali', 'Cham',
    'Chamorro', 'Gujarati', 'Hindi', 'Indonesian', 'Khmer', 'Kmhmu', 'Korean', 'Laotian', 'Malayalam', 'Malay', 'Marathi',
    'Marshallese', 'Nepali', 'Sherpa', 'Tamil', 'Thai', 'Tibetan', 'Trukese', 'Vietnamese', 'Amoy', 'Burmese',
    'Cantonese', 'Chinese', 'Chiu Chow', 'Chow Jo', 'Fukienese', 'Hakka',
    'Hmong', 'Hainanese', 'Japanese', 'Mandarin', 'Mien', 'Shanghainese', 'Taiwanese', 'Taishanese', 'Fijian',
    'Palauan', 'Samoan', 'Tongan', 'Bikol', 'Cebuano', 'Ilocano', 'Ilongo', 'Pampangan', 'Pangasinan', 'Tagalog', 'Visayan'
]

def languages_by_year():
    language_year = []
    for i in range(len(df_main)):
        for lang in target_keywords:
            if lang.lower() in df_main.iloc[i,3]:
                year = pd.to_datetime(df_main.iloc[i,5]).year
                pair_lang_year = (year,lang.lower())
                language_year.append(pair_lang_year)
    language_year = pd.DataFrame(language_year, columns = ['Year', 'Languages'])
    language_year = language_year.groupby(['Year', 'Languages']).size().reset_index(name='count')
    language_year = language_year.pivot_table(index='Year', columns='Languages', values='count', fill_value=0).T
    language_year['Total'] = language_year.sum(axis=1)
    return language_year
        
languages = languages_by_year()

languages = languages.iloc[:, 10:]


#%%
import matplotlib.pyplot as plt

top_20_languages = languages.sort_values(by='Total', ascending=False).head(20).iloc[:, :-1].T

#%%

# Plot the histogram
top_20_languages.plot(kind='line', stacked=False, figsize=(16, 9), linewidth=3)

# Add labels and title
plt.xlabel('')
#plt.ylabel('Number of manuscripts')
plt.title('Language trends in Lingbuzz (2002-2022)', fontsize=18)
plt.legend(fontsize=12)


#%%
#Language proportions

proportion_by_year = languages.iloc[:, 5:-1]
proportion_by_year['Total'] = proportion_by_year.sum(axis=1)

proportion_by_year = proportion_by_year.iloc[:, :-1].div(proportion_by_year['Total'], axis=0) * 100

percentage_std = proportion_by_year.std(axis=1).sort_values(ascending=False).head(10)



#%%

proportion_by_year = proportion_by_year.T
proportion_for_plot = proportion_by_year[list(percentage_std.index)[:10]]

# Plot the histogram
proportion_for_plot.plot(kind='line', stacked=False, figsize=(16, 9), linewidth=3)

# Add labels and title
plt.xlabel('')
#plt.ylabel('Number of manuscripts')
#plt.title('Language trends in Lingbuzz (2002-2022)', fontsize=18)
#plt.legend(fontsize=12)


#%%

# obtain the list of keywords
# import re



# def get_keyword_list():
#     keyword_list = []
#     for row in df_main['Keywords']:
#         ms_keywords = re.split(r', |; ', row)
#         keyword_list += ms_keywords
#     keyword_counts = Counter(keyword_list)
#     keyword_list = [keyword for keyword, count in keyword_counts.items() if count > 20]
#     return keyword_list#list(set(keyword_list))

# keywords = get_keyword_list()
# target_keywords = [ 'phonology', 'morphology', 'syntax', 'semantics']
# keywords = [word for word in keywords if word not in target_keywords]

# list_of_pairs = []

# for i in keywords:
#     for j in keywords:
#         if i != j:
#             pair = frozenset([i, j])
#             list_of_pairs.append(pair)

# list_of_pairs = list(set(list_of_pairs))

# list_of_pairs = [list(i) for i in list_of_pairs] # this should make them tuples

#%%
# counter = []

# for row in df_main['Keywords']:
#     for pair in list_of_pairs:
#         if all(element in row for element in pair):
#             counter.append(pair)

#%%

# counter = [(a, b) for a, b in counter if a not in b and b not in a]

# counts_of_pairs = Counter(counter)



#how to get co-occurrences of features

#look for correlations in the usage of keywords


