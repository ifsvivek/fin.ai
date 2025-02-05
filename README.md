# Fin.ai

Fin.ai is an AI-powered chatbot designed to simplify the loan process for small and medium businesses. By leveraging cutting-edge machine learning models, Fin.ai provides instant financial breakdowns and insights to help users make informed lending decisions.

## Features

-   **Loan Eligibility Prediction**: Uses machine learning to predict whether an individual qualifies for a loan.
-   **Financial Breakdown**: Offers detailed financial analysis and insights.
-   **Chat-Based Interface**: Provides an intuitive chat interface for ease of interaction.
-   **Responsive Design**: Built with modern web technologies for seamless use on all devices.

## Project Structure

-   **app.py**: The main application file that runs the backend server.
-   **Prediction.ipynb**: A Jupyter Notebook showcasing model training and data analysis.
-   **random_forest_model.pkl**: The trained machine learning model used for predictions.
-   **templates/**: Contains HTML files that define various views such as `chat_predict.html`, `chat_business.html`, and `sign_in.html`.
-   **static/**: Houses assets including CSS, JavaScript, images, and vendor libraries.
-   **requirements.txt**: Lists the Python dependencies required for the project.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://your-repository-url.git
    cd Fin.ai
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**

    ```sh
    python app.py
    ```

2. **Access the web interface:**

    Open your browser and navigate to `http://127.0.0.1:5000`.

3. **Interact with the Chatbot:**

    Use the chat interface to ask questions or request financial breakdowns and predictions.

## Acknowledgements

-   Thanks to the creators of the open source project dependencies.
-   Special thanks to the Fin.ai team for their innovative work on this project.
