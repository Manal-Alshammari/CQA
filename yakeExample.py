import pandas as pd
import yake




with open("23.txt") as file:
    forum_posts = file.read()

# forum_posts = pd.read_csv("../input/ForumMessages.csv")
#print(forum_posts)
simple_kwextractor = yake.KeywordExtractor()

type(forum_posts)
post_keywords = simple_kwextractor.extract_keywords(forum_posts)
print(post_keywords)