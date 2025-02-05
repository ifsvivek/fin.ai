from flask import Flask, request, render_template, session, jsonify
import numpy as np
import pandas as pd
import openai
import os
import json
import joblib
import dotenv

dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

loaded_model = joblib.load("random_forest_model.pkl")
print("Model loaded successfully.")

TRAINING_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def chatGPT(text):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": text}],
        max_tokens=4000,
        temperature=0.6,
        stream=False,
    )
    return response.choices.message.content


def get_response(prompt, model="llama3-8b-8192"):
    """
    Generates a chat response using OpenAI's llama3-8b-8192 model based on the given prompt.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a nice loan acceptance prediction and assistant for small business enterprises",
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices.message.content


def get_predict_message(country):
    """
    Generates a custom message prompt for requesting loan information.
    """
    format_str = """
    [
        {
            "myCountry": {
                "organizationName": "",
                "link": ""
            },
            "otherCountry": {
                "organizationName": "",
                "link": "",
                "Country": ""
            }
        },
        {
            "myCountry": {
                "organizationName": "",
                "link": ""
            },
            "otherCountry": {
                "organizationName": "",
                "link": "",
                "Country": ""
            }
        }
    ]
    """
    prompt = (
        "Hi, my country is {}. Kindly create a comprehensive list of places I can look out for to get a good loan "
        "for my small business establishment both in my country and other countries. Make sure you give the countries of "
        "the other countries! Give the answer strictly in this format: {}. Thanks."
    ).format(country, format_str)
    prompt_response = get_response(prompt)
    return prompt, prompt_response


def get_further_response(prediction, question, prev_prompt, prev_response):
    """
    Generates a new prompt based on previous conversation context and a prediction result, then gets a response.
    """
    old = str(prev_prompt) + str(prev_response)
    previous_conv = ""
    rev_old = old[::-1]
    for char in rev_old:
        if len(previous_conv) < 2500:
            previous_conv += char
    final_previous_conv = previous_conv[::-1]
    if prediction == 0:
        add_text = " again congrats on your approved loan"
    elif prediction == 1:
        add_text = " again sorry about the unapproved loan"
    else:
        add_text = ""
    final_previous_conv += add_text
    new_prompt = (
        "Question: "
        + question
        + " | Previous Context: "
        + final_previous_conv
        + " | Instruction: Provide a concise, direct answer within 800 characters."
    )
    further_response = get_response(new_prompt)
    return new_prompt, further_response


def get_business_idea(
    country, country_interest, capital_loan, amount, domain_interest, loan_pay_month
):
    """
    Generates a prompt for business ideas based on user's financial situation and interests.
    """
    format_str = """
    [
        {
            "Business_Idea": "",
            "sector": "",
            "link": ""
        },
        {
            "Business Idea": "",
            "sector": "",
            "link": ""
        }
    ]
    """
    if capital_loan == "capital":
        prompt = (
            "Hi, I'm from {}. Kindly help curate few nice business ideas, the domain sector of the business and like to learn more on the business, "
            "considering that I have a capital of {} US Dollars. My domain of business interest is {} and the country where I want to have my business is {}. "
            "Give the answer strictly in this format: {} Thanks."
        ).format(country, amount, domain_interest, country_interest, format_str)
    elif capital_loan == "loan":
        prompt = (
            "Hi, I'm from {}. Kindly help curate few nice business ideas, the domain sector of the business and like to learn more on the business, "
            "considering that I got a loan of {} US Dollars and I am meant to pay back in {} months time. My domain of business interest is {} and the country where "
            "I want to have my business is {}. Give the answer strictly in this format: {} Thanks."
        ).format(
            country,
            amount,
            loan_pay_month,
            domain_interest,
            country_interest,
            format_str,
        )
    idea_response = get_response(prompt)
    return prompt, idea_response


def get_financial_advice(
    country,
    country_interest,
    description,
    capital_loan,
    amount,
    domain_interest,
    loan_pay_month,
):
    """
    Generates a prompt for obtaining financial advice based on the user's financial status and business interests.
    """
    format_str = """
    {
        "financial_breakdown": "",
        "link": ""
    }
    """
    if capital_loan == "capital":
        prompt = (
            "Hi, I'm from {}. Kindly help curate a comprehensive financial breakdown with link to read more on it, for how I would manage my business "
            "considering that I have a capital of {} US Dollars. My domain of business interest is {}, the description is: {} and the country where I want to have my business is {}. "
            "Make your answer strictly in this format: {}."
        ).format(
            country, amount, domain_interest, description, country_interest, format_str
        )
    elif capital_loan == "loan":
        prompt = (
            "Hi, I'm from {}. Kindly help curate a comprehensive financial breakdown with link to read more on it, for how I would manage my business considering that I got a loan "
            "of {} US Dollars and I am meant to pay back in {} months time. My domain of business interest is {}, the description is: {} and the country where I want to have my business is {}. "
            "Make your answer strictly in this format: {}."
        ).format(
            country,
            amount,
            loan_pay_month,
            domain_interest,
            description,
            country_interest,
            format_str,
        )
    advice_response = get_response(prompt)
    return prompt, advice_response


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/form_predict", methods=["GET", "POST"])
def form_predict():
    return render_template("form_predict.html")


@app.route("/form_business_idea", methods=["GET", "POST"])
def form_business_idea():
    return render_template("form_business_idea.html")


@app.route("/sign_in", methods=["GET", "POST"])
def sign_in():
    return render_template("sign_in.html")


@app.route("/services", methods=["GET", "POST"])
def services():
    return render_template("services.html")


@app.route("/form_financial_advice", methods=["GET", "POST"])
def form_financial_advice():
    return render_template("form_financial_advice.html")


@app.route("/next_session", methods=["GET", "POST"])
def next_session():

    name = request.form["name"].capitalize()
    country = request.form["country"]
    session["name"] = name
    session["country"] = country
    return render_template("services.html", country=country, name=name)


@app.route("/chat_predict", methods=["POST"])
def chat_predict_route():
    """Handle loan prediction requests."""
    try:
        data = [
            [
                int(request.form["depend"]),
                int(request.form["education"]),
                int(request.form["employment"]),
                float(request.form["income"]),
                float(request.form["loan_amount"]),
                float(request.form["loan_term"]),
                float(request.form["score"]),
                float(request.form["resident"]),
                float(request.form["commercial"]),
                float(request.form["luxury"]),
                float(request.form["bank"]),
            ]
        ]

        df = pd.DataFrame(data, columns=TRAINING_COLUMNS)
        prediction = int(loaded_model.predict(df))
        country = session.get("country")
        prompt_text, response = get_predict_message(country)
        session["prediction"] = prediction
        session["chat_response"] = response
        session["chat_prompt"] = prompt_text
        bot_predict_response = json.loads(response)

        return render_template(
            "chat_predict.html",
            pred=prediction,
            name=session.get("name"),
            country=country,
            bot_predict_response=bot_predict_response,
        )

    except Exception as e:
        return str(e), 400


@app.route("/further_predict_chat", methods=["GET", "POST"])
def further_predict_chat():
    pred = session.get("prediction", None)
    bot_predict_prompt = session.get("bot_predict_prompt", None)
    bot_predict_response = session.get("bot_predict_response", None)

    if request.method == "POST":
        predict_question = request.form["question"]
        predict_prompt, predict_response = get_further_response(
            prediction=pred,
            question=predict_question,
            prev_prompt=bot_predict_prompt,
            prev_response=bot_predict_response,
        )
        session["bot_predict_response"] = predict_response
        session["bot_predict_prompt"] = predict_prompt

        return jsonify({"response": predict_response})
    return jsonify({"response": "No question received"})


@app.route("/business_idea", methods=["GET", "POST"])
def business_idea():
    country_interest = request.form["country_interest"].capitalize()
    capital_loan = request.form["capital_loan"]
    amount = request.form["amount"]
    domain_interest = request.form["domain_interest"]
    loan_pay_month = request.form["loan_pay_month"]

    country = session.get("country", None)
    name = session.get("name", None)

    bot_business_prompt, bot_business_response = get_business_idea(
        country=country,
        country_interest=country_interest,
        capital_loan=capital_loan,
        amount=amount,
        domain_interest=domain_interest,
        loan_pay_month=loan_pay_month,
    )

    try:
        bot_business_response = json.loads(bot_business_response)
    except Exception as e:
        return f"Error parsing business idea response: {e}"
    session["bot_business_response"] = bot_business_response
    session["bot_business_prompt"] = bot_business_prompt

    return render_template(
        "chat_business.html",
        name=name,
        country=country,
        bot_business_response=bot_business_response,
    )


@app.route("/further_business_chat", methods=["GET", "POST"])
def further_business_chat():
    bot_business_response = session.get("bot_business_response", None)
    bot_business_prompt = session.get("bot_business_prompt", None)

    if request.method == "POST":
        business_question = request.form["question"]
        business_prompt, business_response = get_further_response(
            prediction="",
            question=business_question,
            prev_prompt=bot_business_prompt,
            prev_response=bot_business_response,
        )
        session["bot_business_response"] = business_response
        session["bot_business_prompt"] = business_question
    return jsonify({"response": business_response})


@app.route("/financial_advice", methods=["GET", "POST"])
def financial_advice():
    country_interest = request.form["country_interest"].capitalize()
    capital_loan = request.form["capital_loan"]
    description = request.form["description"]
    amount = request.form["amount"]
    domain_interest = request.form["domain_interest"]
    loan_pay_month = request.form["loan_pay_month"]

    country = session.get("country", None)
    name = session.get("name", None)

    bot_finance_prompt, bot_finance_response = get_financial_advice(
        country=country,
        country_interest=country_interest,
        description=description,
        capital_loan=capital_loan,
        amount=amount,
        domain_interest=domain_interest,
        loan_pay_month=loan_pay_month,
    )

    try:
        bot_finance_response = json.loads(bot_finance_response)
    except Exception as e:
        return f"Error parsing finance response: {e}"
    session["bot_finance_response"] = bot_finance_response
    session["bot_finance_prompt"] = bot_finance_prompt
    return render_template(
        "chat_finance.html",
        name=name,
        country=country,
        bot_finance_response=bot_finance_response,
    )


@app.route("/further_finance_chat", methods=["GET", "POST"])
def further_finance_chat():
    bot_finance_response = session.get("bot_finance_response", None)
    bot_finance_prompt = session.get("bot_finance_prompt", None)

    if request.method == "POST":
        finance_question = request.form["question"]
        finance_prompt, finance_response = get_further_response(
            prediction="",
            question=finance_question,
            prev_prompt=bot_finance_prompt,
            prev_response=bot_finance_response,
        )
        session["bot_finance_response"] = finance_response
        session["bot_finance_prompt"] = finance_question
    return jsonify({"response": finance_response})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
