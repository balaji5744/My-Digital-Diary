from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
print("VADER:", analyzer.polarity_scores('my crush is eating panipuri'))
