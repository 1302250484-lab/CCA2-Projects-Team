# Name: Abhilasha Kale
# PRN: 1302250348
# Subject: AI ML 1 - CCA2
# Program: Hugging Face Sentiment Monitor (FinTech problem statements)


from transformers import pipeline

# ---------- Hugging Face model----------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# ---------- Problem Statements  ----------
problem_statements = {
    "Problem 1: Payment App Customer Reviews": [
        "The UPI payment was instant and the app is very easy to use.",
        "Money got deducted but the transaction failed. Very frustrating!",
        "Cashback offers are great, I save money every month.",
        "The app keeps crashing during payment. Worst experience.",
        "Customer support solved my issue quickly.",
    ],
    "Problem 2: Stock Market News Headlines": [
        "Sensex hits record high as investors cheer strong quarterly results.",
        "Markets crash as inflation fears rattle investors.",
        "Banking stocks surge after positive RBI announcement.",
        "Company shares plunge after huge losses and fraud allegations.",
        "IT sector shows strong growth and hiring picks up.",
    ],
    "Problem 3: Bank Customer Complaints": [
        "My loan application was rejected without any proper reason.",
        "The bank charged me hidden fees and nobody is helping me.",
        "Net banking is smooth and secure, I am very happy.",
        "Refund is delayed for more than a month, very disappointing.",
        "The branch staff was polite and very helpful.",
    ],
    "Problem 4: Social Media Posts on Digital Lending Apps": [
        "This lending app gave me a loan in just 10 minutes, amazing!",
        "Interest rates are too high, it feels like a scam.",
        "Easy documentation and transparent charges, I trust this app.",
        "They keep harassing me with calls, I hate this app.",
        "Great app for small instant loans during emergencies.",
    ],
}

ALERT_LIMIT = 50


for problem, texts in problem_statements.items():
    print("\n" + "=" * 70)
    print(problem)
    print("=" * 70)

    results = sentiment_model(texts)
    positive_count = 0
    negative_count = 0

    for text, result in zip(texts, results):
        label = result["label"]
        score = round(result["score"] * 100, 1)
        if label == "POSITIVE":
            positive_count += 1
        else:
            negative_count += 1
        print(f"[{label} {score}%] {text}")

    total = len(texts)
    negative_percent = round(negative_count / total * 100, 1)
    print("\nSummary -> Positive:", positive_count,
          "| Negative:", negative_count,
          "| Negative %:", negative_percent)

    if negative_percent > ALERT_LIMIT:
        print("ALERT: Many negative sentiment, need to focus ")
    else:
        print("Status: Sentiment ok.")
