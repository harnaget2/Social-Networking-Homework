

```python
import numpy as np
import pandas as pd
import tweepy
import time
import json
import matplotlib.pyplot as plt
from config_bot import consumer_key, consumer_secret, access_token, token_secret
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

counter = 1
sentiments = []
organizations = ['BBCWorld', 'CBSNews', 'CNN', 'FoxNews', 'nytimes']
final_sentiment = []
```


```python
for orgs in organizations:

    public_tweets = api.user_timeline(orgs, count=100, result_type='recent')
   
    for tweet in public_tweets:

        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = counter
        
        sentiments.append({"Date": tweet["created_at"],
                           "Compound": compound,
                           "Positive": pos,
                           "Negative": neu,
                           "Neutral": neg,
                           "Tweets Ago": counter})
        
        
        
        counter = counter + 1
        
    sentiments_pd = pd.DataFrame.from_dict(sentiments)
    sentiments_pd.to_csv(f'{orgs}_Sentiment_analysis_dataframe.csv', sep='\t')
    
    final_sentiment.append({"Average": np.mean(compound),
                           "Organization": str(orgs)})
    
    plt.plot(np.arange(len(sentiments_pd["Compound"])),
        sentiments_pd["Compound"], marker="o", linewidth=0.5,
        alpha=0.8)

    plt.title(f"Sentiment Analysis of Tweeting for {orgs}")
    plt.ylabel("Tweet Polarity")
    plt.xlabel("Tweets Ago")
    plt.savefig(f'{orgs}_sentiment_request.png')
    plt.show()
        
    sentiments_pd = []
    sentiments = []
    #counter = 1
```


![png](output_2_0.png)



![png](output_2_1.png)



![png](output_2_2.png)



![png](output_2_3.png)



![png](output_2_4.png)



```python
final_sentiment_pd = pd.DataFrame.from_dict(final_sentiment)
final_sentiment_pd
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Average</th>
      <th>Organization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.1280</td>
      <td>CBSNews</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.4215</td>
      <td>CNN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>FoxNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.3182</td>
      <td>nytimes</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(final_sentiment_pd['Organization'], final_sentiment_pd['Average'])

plt.title("Compound Sentiment for Major News Outlets")
plt.ylabel("Average Sentiment")
plt.xlabel("Organization")
plt.savefig('Sentiment_bar_graph.png')
plt.show()
```


![png](output_4_0.png)

