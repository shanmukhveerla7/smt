import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from datetime import datetime
from fpdf import FPDF

# Language translations
LANGUAGES = {
    "en": {
        "title": "🌆 Smart City Assistant",
        "subtitle": "Ask about traffic, energy, environment, and infrastructure.",
        "chat": "🤖 AI Chatbot",
        "traffic": "🚦 Traffic Monitor",
        "energy": "⚡ Energy Tracker",
        "environment": "🌍 Environmental Insights",
        "reports": "📊 City Reports",
        "settings": "⚙️ Settings & Preferences",
        "footer": "© 2025 SmartCity Assistant | Built with ❤️ using Streamlit & Watsonx",
        "save_profile": "Save Profile",
        "generate_ai_report": "Generate AI Report Summary",
        "export_pdf": "📄 Export Report as PDF"
    },
    "es": {
        "title": "🌆 Asistente de Ciudad Inteligente",
        "subtitle": "Pregunte sobre tráfico, energía y medio ambiente.",
        "chat": "🤖 Chatbot con IA",
        "traffic": "🚦 Monitoreo del Tráfico",
        "energy": "⚡ Seguimiento Energético",
        "environment": "🌍 Información Ambiental",
        "reports": "📊 Informes de la Ciudad",
        "settings": "⚙️ Configuración y Preferencias",
        "footer": "© 2025 Asistente de Ciudad Inteligente | Hecho con ❤️ usando Streamlit & Watsonx",
        "save_profile": "Guardar Perfil",
        "generate_ai_report": "Generar Informe con IA",
        "export_pdf": "📄 Exportar Informe como PDF"
    },
    "fr": {
        "title": "🌆 Assistant Ville Intelligent",
        "subtitle": "Posez des questions sur le trafic, l'énergie et l'environnement.",
        "chat": "🤖 Chatbot avec IA",
        "traffic": "🚦 Surveillance du Trafic",
        "energy": "⚡ Suivi Énergétique",
        "environment": "🌍 Analyse Environnementale",
        "reports": "📊 Rapports Urbains",
        "settings": "⚙️ Paramètres et Préférences",
        "footer": "© 2025 Assistant Ville Intelligent | Réalisé avec ❤️ en utilisant Streamlit & Watsonx",
        "save_profile": "Enregistrer le Profil",
        "generate_ai_report": "Générer un Résumé IA",
        "export_pdf": "📄 Exporter le Rapport en PDF"
    }
}

# Page config
st.set_page_config(page_title="🌆 Smart City Assistant", layout="wide", page_icon="🌆")

# Custom CSS - Clean, modern minimalist theme
st.markdown("""
    <style>
        /* Reset */
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background-color: #f9f9f9;
            font-family: 'Helvetica Neue', sans-serif;
            color: #333;
        }

        .main {
            padding: 30px;
            max-width: 1200px;
            margin: auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        }

        h1, h2 {
            color: #2c3e50;
            font-weight: normal;
            font-size: 24px;
        }

        label {
            font-weight: normal;
            color: #34495e;
            font-size: 14px;
        }

        input, select, textarea {
            font-size: 14px;
            font-weight: normal;
        }

        /* Navigation Bar */
        .navbar {
            display: flex;
            justify-content: center;
            gap: 20px;
            padding: 10px 0;
            margin-bottom: 20px;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        .nav-button {
            background-color: #ffffff;
            color: #2c3e50;
            border: 1px solid #ddd;
            width: 140px;
            height: 40px;
            font-size: 14px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .nav-button:hover {
            background-color: #f2f2f2;
            color: #2980b9;
            border-color: #ccc;
        }

        .section-title {
            font-size: 20px;
            font-weight: bold;
            margin-top: 10px;
            color: #2c3e50;
        }

        /* Dotted Line Divider */
        .dotted-line {
            border-top: 1px dotted #ccc;
            margin: 20px 0;
        }

        /* Card Layout */
        .card {
            background-color: #fff;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 14px;
            color: #777;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #eee;
        }

        /* Chat bubbles */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
        }

        .user-bubble, .bot-bubble {
            max-width: 60%;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 14px;
            word-wrap: break-word;
        }

        .user-bubble {
            align-self: flex-end;
            background-color: #2c3e50;
            color: white;
        }

        .bot-bubble {
            align-self: flex-start;
            background-color: #ecf0f1;
            color: black;
        }

        /* Buttons */
        button {
            background-color: #2c3e50;
            color: white;
            border: none;
            padding: 10px 16px;
            font-size: 14px;
            border-radius: 6px;
            cursor: pointer;
        }

        button:hover {
            background-color: #1a252f;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .navbar {
                flex-wrap: wrap;
            }
            .nav-button {
                width: 100%;
                margin: 5px 0;
            }
        }

    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "current_section" not in st.session_state:
    st.session_state.current_section = "chat"
if "language" not in st.session_state:
    st.session_state.language = "en"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "city_data" not in st.session_state:
    st.session_state.city_data = {}

# Load Watsonx credentials
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]
except KeyError as e:
    st.warning(f"⚠️ Missing Watsonx credential: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

model_map = {
    "chat": "ibm/granite-13b-instruct-v2",
    "traffic": "ibm/granite-13b-instruct-v2",
    "energy": "ibm/granite-13b-instruct-v2",
    "environment": "ibm/granite-13b-instruct-v2",
    "reports": "ibm/granite-13b-instruct-v2"
}

def get_llm(model_name):
    return WatsonxLLM(
        model_id=model_map[model_name],
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=project_id,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.TEMPERATURE: 0.7,
            GenParams.MIN_NEW_TOKENS: 5,
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
        },
    )

# Function to export data as PDF including user profile
def export_city_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="SmartCityAI - City Analysis Report", ln=True, align='C')
    pdf.ln(10)
    if "profile_data" in st.session_state and st.session_state.profile_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="User Information", ln=True)
        pdf.set_font("Arial", '', 12)
        for key, value in st.session_state.profile_data.items():
            pdf.cell(0, 10, txt=f"{key.capitalize()}: {value}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Recent City Metrics", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt="Avg Traffic Delay: 12 mins", ln=True)
    pdf.cell(0, 10, txt="Avg CO2 Level: 410 ppm", ln=True)
    pdf.output("city_report.pdf")
    return open("city_report.pdf", "rb").read()

# Navigation Bar
def render_navbar():
    lang = st.session_state.language
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button(LANGUAGES[lang]["chat"], key="btn_chat", use_container_width=True):
            st.session_state.current_section = "chat"
    with col2:
        if st.button(LANGUAGES[lang]["traffic"], key="btn_traffic", use_container_width=True):
            st.session_state.current_section = "traffic"
    with col3:
        if st.button(LANGUAGES[lang]["energy"], key="btn_energy", use_container_width=True):
            st.session_state.current_section = "energy"
    with col4:
        if st.button(LANGUAGES[lang]["environment"], key="btn_environment", use_container_width=True):
            st.session_state.current_section = "environment"
    with col5:
        if st.button("🧾", key="btn_profile", use_container_width=True):
            st.session_state.current_section = "profile"
    with col6:
        if st.button("⚙️", key="btn_settings", use_container_width=True):
            st.session_state.current_section = "settings"
    st.markdown('</div>', unsafe_allow_html=True)

# Header
lang = st.session_state.language
st.markdown(f'<h1 style="text-align:center;">{LANGUAGES[lang]["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:16px;">{LANGUAGES[lang]["subtitle"]}</p>', unsafe_allow_html=True)
render_navbar()

# Functions
def save_profile(name, role, department, location):
    st.session_state.profile_data = {
        "name": name,
        "role": role,
        "department": department,
        "location": location
    }
    st.success("✅ Profile saved successfully!")

def reset_profile():
    st.session_state.profile_complete = False
    st.session_state.profile_data = {}
    st.session_state.messages = []
    st.session_state.city_data = {}
    st.rerun()

# ------------------------------ SETTINGS ------------------------------
if st.session_state.current_section == "settings":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">{LANGUAGES[lang]["settings"]}</h2>', unsafe_allow_html=True)
    language = st.selectbox("Language", options=["en", "es", "fr"], format_func=lambda x: {"en": "English", "es": "Español", "fr": "Français"}[x])
    theme = st.selectbox("Theme", ["Light"])
    font_size = st.slider("Font Size", 12, 24)
    if st.button("💾 Save Preferences"):
        st.session_state.language = language
        st.success("Preferences updated!")
    st.markdown('</div>')

# ------------------------------ USER PROFILE ------------------------------
elif st.session_state.current_section == "profile":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🧾 Complete Your Profile</h2>', unsafe_allow_html=True)
    name = st.text_input("Full Name")
    role = st.selectbox("Role", ["Mayor", "Engineer", "Planner", "Analyst"])
    department = st.text_input("Department")
    location = st.text_input("City / District")
    if st.button("Save Profile"):
        if name and role and department and location:
            save_profile(name, role, department, location)
        else:
            st.error("❌ Please fill in all fields.")
    if "profile_data" in st.session_state and st.session_state.profile_data:
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button("🔄 Reset Profile"):
            reset_profile()
    st.markdown('</div>')
    st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)

# If profile not completed, allow partial access
elif st.session_state.current_section != "profile" and not st.session_state.get("profile_complete", True):
    st.info("ℹ️ You can still explore without completing your profile.")
    if st.button("Go to Profile"):
        st.session_state.current_section = "profile"

# ------------------------------ CHATBOT ------------------------------
elif st.session_state.current_section == "chat":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🤖 AI Chatbot</h2>', unsafe_allow_html=True)
    for role, content in st.session_state.messages:
        bubble_class = "user-bubble" if role == "user" else "bot-bubble"
        st.markdown(f'<div class="{bubble_class}"><b>{role.capitalize()}:</b> {content}</div>', unsafe_allow_html=True)
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Your question:", placeholder="Type something like 'What's the traffic today?'...")
        submit_button = st.form_submit_button(label="Send")
    if submit_button and user_input:
        st.session_state.messages.append(("user", user_input))
        with st.spinner("Thinking..."):
            try:
                llm = get_llm("chat")
                response = llm.invoke(user_input)
                st.session_state.messages.append(("assistant", response))
                st.rerun()
            except Exception as e:
                st.session_state.messages.append(("assistant", f"Error: {str(e)}"))
                st.rerun()
    st.markdown('</div>')
    st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)

# ------------------------------ TRAFFIC MONITOR ------------------------------
elif st.session_state.current_section == "traffic":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🚦 Traffic Monitor</h2>', unsafe_allow_html=True)
    query = st.text_area("Describe your traffic-related issue or question:")
    if st.button("Get Advice"):
        llm = get_llm("traffic")
        res = llm.invoke(query)
        st.markdown(f'<div class="bot-bubble"><b>🧠 AI Response:</b> {res}</div>', unsafe_allow_html=True)
    st.markdown('</div>')
    st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)

# ------------------------------ ENERGY TRACKER ------------------------------
elif st.session_state.current_section == "energy":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">⚡ Energy Tracker</h2>', unsafe_allow_html=True)
    query = st.text_input("Ask about power usage or grid issues:")
    if st.button("Get Suggestions"):
        llm = get_llm("energy")
        res = llm.invoke(query)
        st.markdown(f'<div class="bot-bubble"><b>💡 Suggestion:</b> {res}</div>', unsafe_allow_html=True)
    st.markdown('</div>')
    st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)

# ------------------------------ ENVIRONMENT ANALYSIS ------------------------------
elif st.session_state.current_section == "environment":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🌍 Environmental Insights</h2>', unsafe_allow_html=True)
    query = st.text_area("Ask about pollution, air quality, or sustainability:")
    if st.button("Get Insight"):
        llm = get_llm("environment")
        res = llm.invoke(query)
        st.markdown(f'<div class="bot-bubble"><b>🌱 Analysis:</b> {res}</div>', unsafe_allow_html=True)
    st.markdown('</div>')
    st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)

# ------------------------------ PROGRESS REPORTS ------------------------------
elif st.session_state.current_section == "reports":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">📊 {LANGUAGES[lang]["reports"]}</h2>', unsafe_allow_html=True)
    
    traffic_delay = st.slider("Avg Daily Traffic Delay (min)", 0, 60, step=1)
    co2_level = st.slider("CO2 Level (ppm)", 300, 600, step=5)
    energy_use = st.slider("Energy Use (kWh/day)", 50, 500, step=10)
    waste_ton = st.slider("Waste Collected (ton)", 0, 100, step=1)

    if st.button("Save Data"):
        st.session_state.city_data.update({
            "traffic_delay": traffic_delay,
            "co2_level": co2_level,
            "energy_use": energy_use,
            "waste_ton": waste_ton
        })
        st.success("Data saved successfully.")

    if st.button(LANGUAGES[lang]["generate_ai_report"]):
        summary = get_llm("reports").invoke(f"Give a short city analysis based on: {st.session_state.city_data}")
        st.markdown(f'<div class="bot-bubble"><b>🧠 Summary:</b> {summary}</div>', unsafe_allow_html=True)

    if st.session_state.city_data:
        st.download_button(
            label=LANGUAGES[lang]["export_pdf"],
            data=export_city_report(),
            file_name="city_report.pdf",
            mime="application/pdf"
        )
    st.markdown('</div>')
    st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)

# Footer
st.markdown(f'<p class="footer">{LANGUAGES[lang]["footer"]}</p>', unsafe_allow_html=True)

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
