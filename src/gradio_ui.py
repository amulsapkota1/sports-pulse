import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import file_read
import gradio as gr
import pandas as pd
import json
import re
import openai
import tempfile
import soundfile as sf
import os
from dotenv import load_dotenv

load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

client = openai.OpenAI(
    api_key=OPEN_API_KEY)


SYSTEM_PROMPT = (
    " Answer the question based on the given context. Also return the source URL and any relevant metadata for each part of your answer."
)

# --- Static Data for UI ---
# Quick Facts about Rabindra Dhant and MMA in Nepal
quick_facts_data = """
**Key Highlights:**
*   **Rabindra Dhant is Nepal's first MFN Bantamweight Champion.** He secured this title at Matrix Fight Night 17 in August 2025, defeating India's Chungreng Koren by TKO in the third round.
*   **He famously refused an offer for Indian citizenship** to represent Nepal internationally after winning a national-level amateur MMA championship in India at age 18 in 2019. This decision, made due to Nepal lacking an official MMA association at the time, initially "shattered" his world championship dream, but he remained committed to representing Nepal.
*   **Dhant's early life involved manual labor in India** after leaving his village in Bajhang, Nepal, at 16, where he secretly began martial arts training.
*   **Diwiz Piya Lama is a crucial mentor and coach for Dhant.** Lama, a seasoned Jiu-Jitsu practitioner, met Dhant around 2021 in Kathmandu and has personally funded his training, becoming an "important figure" in his life. Dhant trains at Lock N Roll MMA Nepal and Soma Fight Club in Bali.
*   **MMA's popularity is growing in Nepal.** Dhant's journey and victories are seen as inspiring, generating massive public support and increasing interest in MMA within the country, despite challenges like a lack of infrastructure and government support.
"""

# Sample Prompts for the Chat Interface
sample_prompts_data = """
**Ask me about:**
*   "Tell me about Rabindra Dhant's background and his early struggles."
*   "What was the significance of Rabindra Dhant's refusal of Indian citizenship?"
"""

# Fighter comparison data for table display (attributes as rows, fighters as columns)
fighter_comparison_data = {
    "Attribute": [
        "Nationality",
        "Nickname",
        "Age",
        "Height",
        "Team",
        "Record",
        "Win %",
        "Win Prediction"
    ],
    "Rabindra Dhant": [
        "🇳🇵 Nepalese",
        "The Tiger of Bajhang",
        "26 years, 8 months, 3 days",
        "5'9\" (175cm)",
        "Lock N Roll MMA Nepal / Soma Fight Club Bali",
        "8-1-0",
        "88.9%",
        "🔥 65%"
    ],
    "Chungreng Koren": [
        "🇮🇳 Indian",
        "The Indian Rhino",
        "27 years, 6 months, 1 day",
        "5'8\" (173cm)",
        "Warrior's Cove Mixed Martial Arts",
        "7-1-0",
        "87.5%",
        "⚡ 35%"
    ]
}

# Fight event data
fight_event_data = {
    "event": "Matrix Fight Night 17 (MFN 17)",
    "date": "August 2, 2025",
    "location": "Greater Noida, India",
    "result_method": "KO/TKO (Punches)",
    "result_round": "Round 3",
    "result_time": "0:53",
    "winner": "Rabindra Dhant"
}


def extract_json_from_text(text):
    if not isinstance(text, str):
        return None
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def get_quick_facts():
    return quick_facts_data


def get_sample_prompts():
    return sample_prompts_data


def get_fighter_comparison_table():
    df = pd.DataFrame(fighter_comparison_data)
    return df


def get_fight_event_info():
    data = fight_event_data
    markdown_output = f"""
### 🏆 **{data['event']}** 🏆

**📅 Date:** {data['date']}
**📍 Location:** {data['location']}

**🥊 Fight Result:**
*   **Winner:** **{data['winner']}** 🏆
*   **Method:** {data['result_method']}
*   **Round:** {data['result_round']}
*   **Time:** {data['result_time']}
"""
    return markdown_output


def chat_with_markdown(user_input, history=[], stats_df=None):
    if not user_input or user_input.strip() == "":
      return history, history, stats_df, user_input  # Return current state unchanged

    # Drop rows with missing chunk_text
    df_filtered = file_read.readFile().dropna(subset=["chunk_text"])

    # Create list of texts
    texts = df_filtered["chunk_text"].tolist()

    # Create metadata for each chunk
    metadatas = df_filtered[[
        "source_url",
        "tags"
    ]].to_dict(orient="records")

    # Load a pre-trained embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text to embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    # Create ChromaDB client
    client = chromadb.Client(Settings())

    # Create a collection
    collection_name = "rabindra_info"
    existing_collections = [col.name for col in client.list_collections()]
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)

    # Add data to the collection
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(texts))]
    )


    #embedding user query
    query_embedding = model.encode([user_input])[0]

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas"]
    )

    # Combine text and metadata for prompt
    context_blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        block = f"""
    Source Title: {meta.get("source_title", "N/A")}
    Author: {meta.get("author_or_channel", "N/A")}
    Published Date: {meta.get("published_date", "N/A")}
    Source URL: {meta.get("source_url", "N/A")}
    Tags: {meta.get("tags", "N/A")}
    Entities: {meta.get("entities", "N/A")}

    Content:
    {doc}
    """

    context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    client = openai.OpenAI(
    api_key=OPEN_API_KEY)

    # Prepare messages for LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "Context: " + context}]
    for h in history:
        messages.append({"role": "user", "content": re.sub(r"^👤 ", "", h[0])})
        messages.append({"role": "assistant", "content": re.sub(r"^🤖 ", "", h[1])})
    messages.append({"role": "user", "content": user_input})

    # Call GPT
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    print(f"response: {response.choices}" )
    answer = response.choices[0].message.content or ""

    # Remove JSON from text before display
    #display_text = re.sub(r'\{.*\}', '', answer, flags=re.DOTALL).strip()

    # Wrap user text in white span
    user_display = f"<span style='color: red'>👤</span> <span style='color: #FFFFFF'>{user_input}</span>"

    history.append((user_display, f"🤖 {answer}"))

    return history, history, stats_df, ""



quick_questions = [
        "Tell me about Rabindra Dhant's background",
        "What was the significance of his refusal of Indian citizenship?",
        "Who is his coach and mentor?",
        "How popular is MMA in Nepal?"
]
def send_quick_question(question, chatbot, msg):
    return chat_with_markdown(question, chatbot, msg)  # reuse your existing chat function


def transcribe_audio_to_input(audio_data):
    if not audio_data:
        return ""

    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_array = audio_data

        # Create temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_array, sample_rate)
            temp_file_path = temp_file.name

        try:
            # Transcribe using OpenAI Whisper
            with open(temp_file_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                ).text
            return transcript

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    else:
        return "Invalid audio format"

def toggle_division(is_visible):
                        """Toggle the visibility of the division"""
                        return not is_visible



# Gradio UI
with gr.Blocks(css="""
 #response-box{
}
 #input-box {
     border-radius: 15px !important
}
 .user{
     background: rgb(46 111 156) !important;
}
 #clear{
     font-weight: 600;
     border-radius: 12px;
     border: none;
     padding: 12px 20px;
     cursor: pointer;
     width: 100%;
     margin-top: 10px;
     transition: all 0.3s ease;

 } #send-btn {
     background: linear-gradient(90deg, rgb(54 36 86), rgb(98 162 206)) !important;
     color: white;
     font-weight: 600;
     border-radius: 12px;
     border: none;
     padding: 12px 20px;
     cursor: pointer;
     width: 100%;
     margin-top: 10px;
     transition: all 0.3s ease;
}
 #clear:hover, #send-btn:hover {
     filter: brightness(1.1);
     transform: translateY(-1px);
}
 #fighter-table {
     border: none !important;
     border-radius: 12px !important;
     margin-bottom: 30px !important;
     overflow: hidden !important;
     box-shadow: 0 4px 20px rgba(54, 36, 86, 0.15) !important;
     background: white !important;
}
 #fighter-table table {
     border-collapse: collapse !important;
     width: 100% !important;
     margin: 0 !important;
}
 #fighter-table th, #fighter-table td {
     padding: 12px !important;
     text-align: left !important;
     border: none !important;
     transition: all 0.3s ease !important;
}
 #fighter-table th {
     background: linear-gradient(90deg, rgb(54, 36, 86), rgb(46, 112, 157)) !important;
     font-weight: 700 !important;
     color: white !important;
     text-align: center !important;
     font-size: 14px !important;
     text-transform: uppercase !important;
     letter-spacing: 0.5px !important;
     position: relative !important;
     border-bottom: 2px solid rgba(255, 255, 255, 0.2) !important;
}
 #fighter-table th::after {
     content: '' !important;
     position: absolute !important;
     bottom: 0 !important;
     left: 0 !important;
     right: 0 !important;
     height: 2px !important;
     background: linear-gradient(90deg, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0)) !important;
}
 #fighter-table td:first-child {
     background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
     font-weight: 700 !important;
     color: rgb(54, 36, 86) !important;
     border-right: 3px solid rgba(54, 36, 86, 0.1) !important;
     text-transform: capitalize !important;
     width: 120px !important;
}
 #fighter-table td:not(:first-child) {
     background-color: white !important;
     color: #333 !important;
     border-bottom: 1px solid #f0f0f0 !important;
     font-size: 13px !important;
}
 #fighter-table tr:nth-child(even) td:not(:first-child) {
     background-color: #fafbfc !important;
}
 #fighter-table td:nth-child(2) {
     border-left: 3px solid rgba(54, 36, 86, 0.3) !important;
     padding-left: 15px !important;
}
 #fighter-table td:nth-child(3) {
     border-left: 3px solid rgba(46, 112, 157, 0.3) !important;
     padding-left: 15px !important;
}
 #fighter-table tr:last-child td {
     background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%) !important;
     font-weight: 700 !important;
     font-size: 14px !important;
     border-top: 2px solid #ffd54f !important;
     position: relative !important;
}
 #fighter-table tr:last-child td:first-child {
     background: linear-gradient(135deg, rgb(54, 36, 86), rgb(46, 112, 157)) !important;
     color: white !important;
     border-right: 3px solid #ffd54f !important;
}
 #fighter-table tr:last-child td:not(:first-child) {
     animation: subtle-glow 3s ease-in-out infinite alternate !important;
}
 @keyframes subtle-glow {
     from {
         box-shadow: inset 0 0 5px rgba(255, 193, 7, 0.3) !important;
    }
     to {
         box-shadow: inset 0 0 15px rgba(255, 193, 7, 0.5) !important;
    }
}
 #fighter-table tr:not(:first-child):hover td {
     transform: translateY(-1px) !important;
     box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}
 #fighter-table tr:hover td:first-child {
     background: linear-gradient(135deg, rgb(54, 36, 86), rgb(46, 112, 157)) !important;
     color: white !important;
}
 #fighter-table tr:not(:first-child):hover td:not(:first-child) {
     background-color: #f0f7ff !important;
}
 @media (max-width: 768px) {
     #fighter-table th, #fighter-table td {
         padding: 8px 6px !important;
         font-size: 12px !important;
    }
     #fighter-table td:first-child {
         width: 100px !important;
    }
}
.audio_upload_area, .input_audio_upload, .audio_upload_dropzone {
    display: none !important;
}

#micphone {
width: 100% !important;
}

""") as demo:
    gr.Markdown("## MMS Chat Assistant")

    with gr.Row():
        with gr.Column(scale=3):
            # Chatbot
            chatbot = gr.Chatbot(elem_classes="chat-box", elem_id="response-box")

            # Quick question buttons above input
            with gr.Row():
                buttons = []
                for q_text in quick_questions:
                    btn = gr.Button(q_text, elem_classes="quick-btn")
                    buttons.append(btn)

            # User input textbox
            msg = gr.Textbox(label="Enter your message", placeholder="Type your question here...", elem_id="input-box")

            with gr.Row():
                with gr.Column(scale=2):
                    clear = gr.Button("Clear Chat", elem_id="clear", scale=3)

                with gr.Column(scale=2):
                    send_btn = gr.Button("Send", elem_id="send-btn")

                    microphone_toggle = gr.Button("🎤")

                    # Division that will be shown/hidden
                    with gr.Column(visible=False) as division:
                        speech_input = gr.Audio(label="🎤 Record your question", sources=["microphone"], type="numpy",
                                                interactive=True, elem_id="micphone")
                    # State to track division visibility
                    division_visible = gr.State(False)

                    # Event handler
                    microphone_toggle.click(
                        fn=toggle_division,
                        inputs=[division_visible],
                        outputs=[division_visible]
                    ).then(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[division_visible],
                        outputs=[division]
                    ).then(
                        fn=lambda x: "🔴" if x else "🎤",
                        inputs=[division_visible],
                        outputs=[microphone_toggle]
                    )

            # Connect quick question buttons to chatbot
            for btn, q_text in zip(buttons, quick_questions):
                btn.click(
                    send_quick_question,
                    inputs=[gr.State(q_text), chatbot, msg],
                    outputs=[chatbot, chatbot, msg, msg]
                )

        # Right sidebar with fighter info
        with gr.Column(scale=1):
            gr.Markdown("## 🥊 Fighter Comparison & Prediction")
            fighter_table = gr.DataFrame(
                value=get_fighter_comparison_table(),
                elem_id="fighter-table",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("## 💡 Quick Facts & Context")
            gr.Markdown(get_quick_facts())

    # Submit textbox on Enter
    msg.submit(
        chat_with_markdown,
        inputs=[msg, chatbot],
        outputs=[chatbot, chatbot, msg]
    )

    # Clear chat
    clear.click(
        lambda: ([], [], get_fighter_comparison_table(), ""),
        None,
        [chatbot, chatbot, fighter_table, msg],
        queue=False
    )

    send_btn.click(chat_with_markdown, inputs=[msg, chatbot], outputs=[chatbot, chatbot, msg])

    speech_input.change(transcribe_audio_to_input, inputs=[speech_input], outputs=[msg])


def main():
 demo.launch()