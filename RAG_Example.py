# Name: Aditya Gupta
# PRN: 1302250484
# Subject: AI ML 1 - CCA2
# Program: Retrieval Augmented Generation (RAG) - FinTech Help Bot


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Knowledge Base (hamare documents) ----------
knowledge_base = [
    {"source": "UPI Guide",
     "text": "UPI stands for Unified Payments Interface. It allows instant money transfer between bank accounts using a mobile app 24 hours a day."},
    {"source": "Credit Score Guide",
     "text": "A credit score is a three digit number between 300 and 900. A score above 750 is considered good and increases the chance of loan approval."},
    {"source": "Loan Guide",
     "text": "A loan is money borrowed from a bank that must be repaid with interest. The bank checks income and credit score before loan approval."},
    {"source": "KYC Guide",
     "text": "KYC means Know Your Customer. Banks and fintech apps verify identity using documents like Aadhaar and PAN card before opening an account."},
    {"source": "Mutual Fund Guide",
     "text": "A mutual fund collects money from many investors and invests it in stocks and bonds. A SIP lets you invest a small fixed amount every month."},
    {"source": "Digital Wallet Guide",
     "text": "A digital wallet stores money in an app so you can pay online quickly. Examples include Paytm and PhonePe wallets."},
]

# ---------- RULES ----------
SIMILARITY_LIMIT = 0.15      # Rule 4
TOP_K = 2                    # Rule 6
SENSITIVE_WORDS = ["password", "otp", "pin", "cvv", "card number"]
ADVICE_WORDS = ["should i buy", "should i invest", "which stock", "best stock", "guaranteed return"]


documents = [item["text"] for item in knowledge_base]
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(documents)

def rag_answer(question):
    q = question.strip()
    q_lower = q.lower()

    # Rule 1
    if len(q) < 5:
        return "Rule 1: Please ask complete question (atleast5 characters)."

    # Rule 2
    for word in SENSITIVE_WORDS:
        if re.search(r"\b" + word + r"\b", q_lower):
            return "Rule 2: For security reasons, I do not share or ask for information such as passwords, OTPs, PINs, or CVVs."

    # Retrieval
    q_vector = vectorizer.transform([q])
    scores = cosine_similarity(q_vector, doc_vectors)[0]
    top_indexes = scores.argsort()[::-1][:TOP_K]   # Rule 6

    # Rule 4
    if scores[top_indexes[0]] < SIMILARITY_LIMIT:
        return "Rule 4: Sorry, I don't have any information regarding this."

    # Generation
    answer = ""
    sources = []
    for i in top_indexes:
        if scores[i] >= SIMILARITY_LIMIT:
            answer += knowledge_base[i]["text"] + " "
            sources.append(knowledge_base[i]["source"])

    # Rule 5
    answer = answer.strip() + "\n   Source: " + ", ".join(sources)

    # Rule 3
    for phrase in ADVICE_WORDS:
        if phrase in q_lower:
            answer += "\n   Rule 3 Disclaimer: Its not financial advice. Please seek advise from advisor before investing"
            break

    # Rule 7
    answer += f"\n   Rule 7 Confidence: {round(scores[top_indexes[0]] * 100, 1)}%"
    return answer

# ---------- Test sawaal ----------
questions = [
    "What is UPI?",
    "How does credit score affect loan approval?",
    "What is KYC and which documents are needed?",
    "Tell me about SIP in mutual fund",
    "Should I invest in mutual fund for guaranteed return?",
    "What is my bank password?",
    "Who won the cricket world cup?",
    "hi",
]

for question in questions:
    print("\nQuestion:", question)
    print("Answer  :", rag_answer(question))
