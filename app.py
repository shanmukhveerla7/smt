import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from datetime import datetime
from fpdf import FPDF

# Language translations for smart city domain
LANGUAGES = {
    "en": {
        "title": "🌆 Smart City Assistant",
        "subtitle": "Ask about traffic, energy, environment, and infrastructure.",
        "home_welcome": "🌆 Welcome to Your Smart City Assistant",
        "highlights": "### 🏙️ Highlights:",
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
        "home_welcome": "🌆 Bienvenido al Asistente de Ciudad Inteligente",
        "highlights": "### 🏙️ Destacados:",
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
        "home_welcome": "🌆 Bienvenue dans votre Assistant Ville Intelligent",
        "highlights": "### 🏙️ Points forts :",
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

# Custom CSS - Urban Blue Theme
st.markdown("""
    <style>
        body { background-color: #e8f4ff; font-family: 'Segoe UI', sans-serif; }
        .main { background-color: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .card { background-color: #ffffff; padding: 25px; margin: 20px 0; border-left: 6px solid #3498db; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .navbar { display: flex; justify-content: center; gap: 20px; padding: 15px 0; background: linear-gradient(to right, #3498db, #2ecc71); border-radius: 10px; margin-bottom: 25px; }
        .nav-button { background-color: #ffffff; color: #3498db; border: none; width: 50px; height: 50px; font-size: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.3s ease; }
        .nav-button:hover { background-color: #def8ff; transform: scale(1.1); }
        h1, h2, h3 { color: #2c3e50; }
        label { font-weight: bold; color: #34495e; }
        input, select, textarea { border-radius: 8px; border: 1px solid #ccc; padding: 10px; width: 100%; font-size: 14px; }
        button { background-color: #3498db; color: white; border: none; padding: 10px 20px; font-size: 14px; border-radius: 8px; cursor: pointer; }
        button:hover { background-color: #2980b9; }
        .user-bubble, .bot-bubble { padding: 10px 15px; border-radius: 12px; max-width: 70%; margin: 6px 0; font-size: 14px; }
        .user-bubble { background-color: #d6ecf3; align-self: flex-end; }
        .bot-bubble { background-color: #dff0d8; align-self: flex-start; }
        .chat-container { display: flex; flex-direction: column; gap: 10px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "profile_complete" not in st.session_state:
    st.session_state.profile_complete = False
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}
if "current_section" not in st.session_state:
    st.session_state.current_section = "profile"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "city_data" not in st.session_state:
    st.session_state.city_data = {}
if "language" not in st.session_state:
    st.session_state.language = "en"

# Load Watsonx credentials
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]
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
except KeyError:
    st.warning("⚠️ Watsonx credentials missing.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

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
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        if st.button("챗", key="btn_chat", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "chat"
    with col2:
        if st.button("🚦", key="btn_traffic", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "traffic"
    with col3:
        if st.button("⚡", key="btn_energy", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "energy"
    with col4:
        if st.button("🌍", key="btn_environment", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "environment"
    with col5:
        if st.button("📊", key="btn_reports", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "reports"
    with col6:
        if st.button("🧾", key="btn_profile", use_container_width=True):
            st.session_state.current_section = "profile"
    with col7:
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
    st.session_state.profile_complete = True
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
    st.markdown(f'<h2>⚙️ {LANGUAGES[lang]["settings"]}</h2>', unsafe_allow_html=True)
    language = st.selectbox("Language", options=["en", "es", "fr"], format_func=lambda x: {"en": "English", "es": "Español", "fr": "Français"}[x])
    theme = st.selectbox("Theme", ["Light"])
    font_size = st.slider("Font Size", 12, 24)
    if st.button(LANGUAGES[lang]["save_profile"]):
        st.session_state.language = language
        st.success("Preferences updated!")
    st.markdown('</div>')

# ------------------------------ USER PROFILE ------------------------------
elif st.session_state.current_section == "profile":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🧾 Complete Your Profile</h2>', unsafe_allow_html=True)
    name = st.text_input("Full Name")
    role = st.selectbox("Role", ["Mayor", "Engineer", "Planner", "Analyst"])
    department = st.text_input("Department")
    location = st.text_input("City / District")
    if st.button("Save Profile"):
        if name and role and department and location:
            save_profile(name, role, department, location)
        else:
            st.error("❌ Please fill in all fields.")
    if st.session_state.profile_complete:
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button("🔄 Reset Profile"):
            reset_profile()
    st.markdown('</div>')

# If profile not completed, stop further access
elif not st.session_state.profile_complete:
    st.info("ℹ️ Please complete your profile before continuing.")
    if st.button("Go to Profile"):
        st.session_state.current_section = "profile"
    st.stop()

# ------------------------------ CHATBOT ------------------------------
elif st.session_state.current_section == "chat":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🤖 AI Chatbot</h2>', unsafe_allow_html=True)
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

# ------------------------------ TRAFFIC MONITOR ------------------------------
elif st.session_state.current_section == "traffic":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🚦 Traffic Monitor</h2>', unsafe_allow_html=True)
    query = st.text_area("Describe your traffic-related issue or question:")
    if st.button("Get Advice"):
        llm = get_llm("traffic")
        res = llm.invoke(query)
        st.markdown(f"🧠 **AI Response:**\n{res}")
    st.markdown('</div>')

# ------------------------------ ENERGY TRACKER ------------------------------
elif st.session_state.current_section == "energy":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>⚡ Energy Tracker</h2>', unsafe_allow_html=True)
    query = st.text_input("Ask about power usage or grid issues:")
    if st.button("Get Suggestions"):
        llm = get_llm("energy")
        res = llm.invoke(query)
        st.markdown(f"💡 **Suggestions:**\n{res}")
    st.markdown('</div>')

# ------------------------------ ENVIRONMENT ANALYSIS ------------------------------
elif st.session_state.current_section == "environment":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🌍 Environmental Insights</h2>', unsafe_allow_html=True)
    query = st.text_area("Ask about pollution, air quality, or sustainability:")
    if st.button("Get Insight"):
        llm = get_llm("environment")
        res = llm.invoke(query)
        st.markdown(f"🌱 **Analysis:**\n{res}")
    st.markdown('</div>')

# ------------------------------ PROGRESS REPORTS ------------------------------
elif st.session_state.current_section == "reports":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2>📊 {LANGUAGES[lang]["reports"]}</h2>', unsafe_allow_html=True)
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
        st.markdown(f"🧠 **AI Analysis:**\n{summary}")
    if st.session_state.profile_complete and st.session_state.city_data:
        st.download_button(
            label=LANGUAGES[lang]["export_pdf"],
            data=export_city_report(),
            file_name="city_report.pdf",
            mime="application/pdf"
        )
    st.markdown('</div>')

# Footer
st.markdown(f'<p style="text-align:center; font-size:14px;">{LANGUAGES[lang]["footer"]}</p>', unsafe_allow_html=True)

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
