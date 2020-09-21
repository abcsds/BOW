import spacy
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups


newsgroups = fetch_20newsgroups(subset='train')
nlp = spacy.load("en_core_web_sm")
wc = WordCloud(
    max_words=100,
    width=2000,
    height=1200,
)

cloud = wc.generate("\n\n".join(newsgroups.data))

plt.imshow(cloud)
plt.axis("off")

cloud = wc.generate("\n\n".join(newsgroups.target_names))
plt.imshow(cloud)
plt.axis("off")

print(list(newsgroups.target_names))

vocab = set()
for text in newsgroups.data:
    doc = nlp(text)
    vocab = vocab.union({token.text for token in doc})
print(len(vocab))

plt.imshow(wc.generate(text))
plt.axis("off")


def unzip(zipped):
    x = [i for i, j in zipped]
    y = [j for i, j in zipped]
    return x, y

i = 0
for text, label in zip(newsgroups.data, newsgroups.target):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    # tokens = [ent.text for ent in doc.ents]
    hist = Counter(tokens)
    words, freqs = unzip(hist.most_common(20))

    plt.figure()
    plt.bar(words, freqs)
    plt.title(newsgroups.target_names[label])
    plt.xticks(rotation=90)
    i += 1
    if i > 10:
        break

len(newsgroups.data)


from sklearn.feature_extraction.text import CountVectorizer
corpus = newsgroups.data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X.toarray())
print(X.shape)

Y = newsgroups.target

print(Y.shape)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

plt.figure()
tree.plot_tree(clf)
plt.show()

newsgroups_test = fetch_20newsgroups(subset='test')
X_test = vectorizer.fit_transform(newsgroups_test.data)
Y_test = newsgroups_test.target

Y_pred = clf.predict(X_test)
Y_pred

from sklearn.metrics import accuracy_score

accuracy_score(Y_test,Y_pred)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X, Y)
Y_pred = clf.predict(X_test)
accuracy_score(Y_test,Y_pred)


# N-grams
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
