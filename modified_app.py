import streamlit as st
from transformers import pipeline

# Load your trained model and tokenizer for inference
model_path = "my_ner_model"  # Update this path to your trained model
ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path)

# Define label mapping for CoNLL-2003 dataset
label_mapping = {
    0: "O",        # Outside any named entity
    1: "B-PER",    # Beginning of a person entity
    2: "I-PER",    # Inside a person entity
    3: "B-ORG",    # Beginning of an organization entity
    4: "I-ORG",    # Inside an organization entity
    5: "B-LOC",    # Beginning of a location entity
    6: "I-LOC",    # Inside a location entity
    7: "B-MISC",   # Beginning of a miscellaneous entity (if applicable)
    8: "I-MISC",   # Inside a miscellaneous entity (if applicable)
}

# Function to convert label IDs to their corresponding names
def convert_label_id_to_name(label_id):
    return label_mapping.get(label_id, "Unknown")

# Configure Streamlit for multi-page layout
st.set_page_config(layout="wide")

# Set background color to light beige and customize boxes
st.markdown("""
    <style>
        body {
            background-color: #f5f5dc;
        }
        .stTextArea textarea {
            background-color: #e6e6fa;
            color: black;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Display author name in top right corner
st.markdown("""
    <div style='position: absolute; top: 10px; right: 20px; font-size: 16px; font-weight: bold;'>
        by Hareetima Sonkar
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["NER App", "About Project"])

if page == "NER App":
    st.title("🌟 Named Entity Recognition (NER) Web App")
    st.write("Enter text below and click 'Submit' to perform NER.")

    # Initialize session state for input_text if not already set
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    # Suggested sentences for quick testing
    suggestions = [
        "Elon Musk is the CEO of Tesla.",
        "Barack Obama was the 44th President of the United States.",
        "Google was founded in California.",
        "The Eiffel Tower is located in Paris, France.",
        "Apple Inc. announced a new iPhone in September."
    ]

    col1, col2, col3, col4, col5 = st.columns(5)
    for i, suggestion in enumerate(suggestions):
        with [col1, col2, col3, col4, col5][i]:
            if st.button(suggestion):
                st.session_state["input_text"] = suggestion  # Update session state

    # Input text area for user input
    input_text = st.text_area("✍️ Enter your text:", height=150, value=st.session_state["input_text"])

    # Button to trigger NER processing
    if st.button("🚀 Submit"):
        if st.session_state["input_text"]:
            # Perform NER on the input text
            results = ner_pipeline(st.session_state["input_text"])

            # Display results with corresponding labels, filtering out specific labels
            st.subheader("📌 NER Results:")
            for entity in results:
                label_name = convert_label_id_to_name(int(entity['entity'].split('_')[-1]))
                if label_name not in ["O", "B-PER"]:  # Exclude 'O' and 'B-PER' labels from output
                    st.markdown(
                        f"""
                        <p style='background-color:#FAD02E; padding:5px; border-radius:5px;'>
                            <b>Entity:</b> {entity['word']} | <b>Type:</b> {label_name} | 
                            <b>Confidence:</b> {entity['score']:.4f}
                        </p>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.warning("Please enter some text for NER.")

elif page == "About Project":
    st.title("📖 About This Project")
    st.write("""
        This Named Entity Recognition (NER) web app is built using a fine-tuned transformer model trained on the CoNLL-2003 dataset.

        ## 📌 Workflow
        
        ### 1️⃣ Data & Model Preparation
        - 📂 **Dataset Used**: CoNLL-2003 (a standard dataset for NER tasks).
        - 🧠 **Model**: Fine-tuned **BERT-based transformer** model for high-accuracy entity recognition.
        - 🔧 **Training Process**:
          - Preprocessing of text data.
          - Tokenization using BERT tokenizer.
          - Fine-tuning on NER-specific data.
          - Evaluation and optimization.
        
        ### 2️⃣ App Features & Functionality
        - 🏷 **Recognizes Named Entities**:
          - **Persons (PER)** → Names of individuals.
          - **Organizations (ORG)** → Companies, institutions, etc.
          - **Locations (LOC)** → Cities, countries, geographical places.
          - **Miscellaneous (MISC)** → Other entity types.
        - ⚡ **Real-time inference with a user-friendly interface.**
        - 📊 **Confidence scores** for detected entities.
        
        ### 3️⃣ How to Use the App
        ✅ **Step 1**: Enter text manually or select a **predefined example**.
        ✅ **Step 2**: Click **Submit** to analyze the text.
        ✅ **Step 3**: View **recognized named entities** with labels and confidence scores.
        
        ### 4️⃣ Behind-the-Scenes Processing
        1️⃣ **Text input is tokenized** using the **BERT tokenizer**.
        2️⃣ **NER Model predicts entity types** for each token.
        3️⃣ **Post-processing** aligns predictions with words.
        4️⃣ **Results are displayed** with confidence scores.
        
        ### 5️⃣ Project Objective
        🚀 **Why This App?**
        This project is part of a **broader study in NLP** aimed at improving **Named Entity Recognition** using deep learning models. It demonstrates the power of **transformer-based models** for extracting meaningful information from text.
    """)
