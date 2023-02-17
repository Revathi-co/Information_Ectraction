import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

comment_words = " ".join(['scrum master', 'software development', 'team', 'project manager', 'delivery', 'role', 'scrum', 'concept', 'agile', 'delivery system', 'vsts', 'ado', 'skill', 'knowledge',  'leadership',  'conflict resolution', 'continuous improvement', 'empowerment',  'transparency', 'certification', 'spc','rte', 'ssm'])
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()