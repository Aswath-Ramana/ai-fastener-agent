# 🔩 AI Fastener Agent

An intelligent Streamlit web application designed to automate the matching of customer fastener orders against a master product catalog. This tool leverages a hybrid AI approach, combining fast local search with the advanced reasoning capabilities of a cloud-based AI Assistant to ensure high accuracy and efficiency.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-streamlit-app-url.streamlit.app/](https://ai-fastener-agent.streamlit.app/))  

![image](https://github.com/user-attachments/assets/eb522873-afdd-4323-b3dc-e1944da6993a)

![image](https://github.com/user-attachments/assets/a8464f1c-8d83-48ec-8c78-f85aae5f9ab4)

---

## ✨ Features

This application provides a powerful two-pronged solution for fastener management:

### 🤖 1. Automated Order Matcher (Bulk Processing)
- **Upload & Go:** Users can upload an Excel file containing a list of order items (part numbers, descriptions, dimensions).
- **Hybrid AI Matching:**
    - **Fast Local Search:** For each order item, the system first performs a high-speed local search using **FAISS (semantic search)** and **Rapidfuzz (fuzzy search)** to identify a shortlist of the top 5 potential candidates from the master data.
    - **AI-Powered Decision Making:** This shortlist is then sent to a pre-configured **OpenAI Assistant** which uses its deep knowledge of the full product catalog and master data to make the final, reasoned decision.
- **Structured Output:** The Assistant determines the single best match or flags the item as a "New Item" if no suitable match is found.
- **Downloadable Reports:** The final results, including the original order details and the AI's match decision, are compiled into a downloadable Excel report for procurement teams.

### 💬 2. Fastener Finder Assistant (Conversational AI)
- **Interactive Chatbot:** A dedicated chat interface allows users to ask ad-hoc questions about products in natural language.
- **Deep Knowledge Access:** The chatbot is directly powered by the same OpenAI Assistant, giving it access to the entire uploaded product catalog and master data files.
- **Technical & Accurate Responses:** The Assistant is instructed to act as a technical fastener expert, providing precise and informative answers based on its knowledge base.
- **Conversational Memory:** The Assistant maintains the context of the conversation, allowing for natural follow-up questions.

---

## 🛠️ Technology Stack

This project is built with a modern stack designed for AI-powered web applications:

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend Logic:** Python
- **AI & Machine Learning:**
    - **Cloud AI:** [OpenAI Assistants API (GPT-4o)](https://platform.openai.com/docs/assistants/overview)
    - **Local Search:**
        - `FAISS` (from Facebook AI) for high-speed vector similarity search.
        - `sentence-transformers` for creating text embeddings.
        - `rapidfuzz` for efficient fuzzy string matching.
- **Data Handling:** `pandas`, `pyarrow`, `openpyxl`
- **Deployment:** Hosted on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 🚀 Getting Started

To run this application, you will need to set up the necessary local assets and configure your AI Assistant.

### Prerequisites

- Python 3.9+
- An [OpenAI API Key](https://platform.openai.com/account/api-keys)
- A GitHub account

### 1. Prepare Local Search Assets

The bulk matching feature relies on a pre-built local search index.

1.  Place your master product list (e.g., `master_data.xlsx`) in a `/data` directory.
2.  Run the preparation script to create the FAISS index and metadata file:
    ```bash
    pip install -r requirements.txt
    python prepare_master_data.py
    ```
3.  This will generate a `faiss_index.bin` and `master_metadata.parquet` file. Move these into a `local_search_assets/` directory in the project root.

### 2. Configure the OpenAI Assistant

The AI's reasoning power comes from a cloud-based Assistant.

1.  **Prepare Knowledge Files:**
    - Run the `convert_master_for_assistant.py` script to convert your `master_data.xlsx` into a clean `assistant_knowledge_base.txt` file.
    - Have your detailed product catalog (e.g., `catalog.pdf`) ready.
2.  **Create the Assistant on OpenAI:**
    - Go to the [OpenAI Assistants platform](https://platform.openai.com/assistants).
    - Create a new Assistant, selecting the `gpt-4o` model.
    - Provide it with detailed instructions on its persona and tasks (see `instructions.md` for a template).
    - Enable the **Retrieval** tool.
    - Upload your `assistant_knowledge_base.txt` and `catalog.pdf` files to its knowledge base.
    - Save the Assistant and copy its **Assistant ID** (e.g., `asst_...`).

### 3. Running the App Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-fastener-agent.git
    cd ai-fastener-agent
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    - Create a `.env` file in the root directory.
    - Add your OpenAI API key to this file:
      ```
      OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
      ```
4.  **Update the Assistant ID:**
    - Open `app.py` and replace the placeholder `ASSISTANT_ID` with the actual ID you copied from the OpenAI platform.
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## ☁️ Deployment

This application is designed for easy deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).

1.  Push your final project code to a public GitHub repository. Ensure your `.gitignore` file prevents secrets (`.env`) and large data files from being committed.
2.  Connect your GitHub account to Streamlit Community Cloud.
3.  Select your repository and deploy.
4.  In the "Advanced settings," add your `OPENAI_API_KEY` as a secret.

The app will be live and accessible to multiple users.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/ai-fastener-agent/issues).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
