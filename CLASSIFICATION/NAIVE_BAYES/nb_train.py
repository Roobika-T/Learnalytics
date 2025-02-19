import re
from collections import defaultdict
import pickle

class NaiveBayes:
    def __init__(self):
        self.good_words = defaultdict(int)
        self.spam_words = defaultdict(int)
        self.total_good_words = 0
        self.total_spam_words = 0

    def train_model(self, good_emails, spam_emails):
        for email in good_emails:
            words = re.findall(r'\b\w+\b', email.lower())
            for word in words:
                self.good_words[word] += 1
                self.total_good_words += 1

        for email in spam_emails:
            words = re.findall(r'\b\w+\b', email.lower())
            for word in words:
                self.spam_words[word] += 1
                self.total_spam_words += 1

    def calculate_likelihood(self):
        likelihood_table = {}
        for word in set(list(self.good_words.keys()) + list(self.spam_words.keys())):
            good_likelihood = (self.good_words[word] + 1) / (self.total_good_words + len(self.good_words))
            spam_likelihood = (self.spam_words[word] + 1) / (self.total_spam_words + len(self.spam_words))
            likelihood_table[word] = [good_likelihood, spam_likelihood]
        return likelihood_table

    def classify_email(self, email):
        words = re.findall(r'\b\w+\b', email.lower())
        good_prob = 1.0
        spam_prob = 1.0

        for word in words:
            if word in self.good_words:
                good_prob *= (self.good_words[word] + 1) / (self.total_good_words + len(self.good_words))
            else:
                good_prob *= 1 / (self.total_good_words + len(self.good_words))  # Unknown word handling
            
            if word in self.spam_words:
                spam_prob *= (self.spam_words[word] + 1) / (self.total_spam_words + len(self.spam_words))
            else:
                spam_prob *= 1 / (self.total_spam_words + len(self.spam_words))  # Unknown word handling

        # Normalize probabilities
        total_prob = good_prob + spam_prob
        good_prob /= total_prob
        spam_prob /= total_prob

        print(f"Good Email Probability: {good_prob:.4f}, Spam Email Probability: {spam_prob:.4f}")
        return f"Good Email Probability: {good_prob:.4f}, Spam Email Probability: {spam_prob:.4f}"

# Data preprocessing
def read_emails_from_file(filename):
    with open(filename, 'r') as file:
        emails = [line.strip() for line in file.readlines()]
    return emails

# Training process
good_emails = read_emails_from_file('Datasets/good_emails.txt')
spam_emails = read_emails_from_file('Datasets/spam_emails.txt')

nb = NaiveBayes()
nb.train_model(good_emails, spam_emails)
likelihood_table = nb.calculate_likelihood()

# Save model and likelihood table as .pkl
with open('Saved_models/naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump({'model': nb, 'likelihood_table': likelihood_table}, model_file)

print("Naive Bayes model saved successfully.")
