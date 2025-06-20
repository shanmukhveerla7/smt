import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from datetime import datetime
from fpdf import FPDF
import requests  # For weather API

# Language translations
LANGUAGES = {
    "en": {
        "title": "🌆 Smart City Assistant",
        "subtitle": "Ask about traffic, energy, environment, weather, and infrastructure.",
        "chat": "🤖 AI Chatbot",
        "traffic": "🚦 Traffic Monitor",
        "energy": "⚡ Energy Tracker",
        "environment": "🌍 Environmental Insights",
        "weather": "🌦️ Weather Forecast",
        "reports": "📊 City Reports",
        "settings": "⚙️ Settings & Preferences",
        "footer": "© 2025 SmartCity Assistant | Built with ❤️ using Streamlit & Watsonx",
    },
    # You can keep other languages if needed
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
        .user-bubble {
            background-color: #2c3e50;
            color: white;
            align-self: flex-end;
            border-radius: 12px;
            padding: 10px 15px;
            max-width: 70%;
            margin: 6px 0;
            font-size: 14px;
        }
        .bot-bubble {
            background-color: #ffffff;
            color: black;
            align-self: flex-start;
            border-radius: 12px;
            padding: 10px 15px;
            max-width: 70%;
            margin: 6px 0;
            font-size: 14px;
        }
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
except KeyError as e:
    st.warning(f"⚠️ Watsonx credential missing: {str(e)}")
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

# Fetch weather data
def get_weather(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
    response = requests.get(complete_url).json()
    if response.get("cod") != 200:
        return None
    main_data = response["main"]
    weather_data = response["weather"][0]
    return {
        "city": city,
        "temp": main_data["temp"],
        "feels_like": main_data["feels_like"],
        "humidity": main_data["humidity"],
        "wind_speed": response["wind"]["speed"],
        "description": weather_data["description"].capitalize(),
    }

# Navigation Bar
def render_navbar():
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        if st.button("챗", key="btn_chat", disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "chat"
    with col2:
        if st.button("🚦", key="btn_traffic", disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "traffic"
    with col3:
        if st.button("⚡", key="btn_energy", disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "energy"
    with col4:
        if st.button("🌍", key="btn_environment", disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "environment"
    with col5:
        if st.button("🌦️", key="btn_weather", disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "weather"
    with col6:
        if st.button("🧾", key="btn_profile"):
            st.session_state.current_section = "profile"
    with col7:
        if st.button("⚙️", key="btn_settings"):
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

# ------------------------------ WEATHER MODULE ------------------------------
if st.session_state.current_section == "weather":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🌦️ Weather Forecast</h2>', unsafe_allow_html=True)
    city = st.text_input("Enter City Name")
    if st.button("Get Weather"):
        if city:
            try:
                weather = get_weather(city, st.secrets["OPENWEATHER_APIKEY"])
                if weather:
                    st.write(f"""
                        **City:** {weather['city']}  
                        **Temperature:** {weather['temp']}°C  
                        **Feels Like:** {weather['feels_like']}°C  
                        **Humidity:** {weather['humidity']}%  
                        **Wind Speed:** {weather['wind_speed']} m/s  
                        **Description:** {weather['description']}
                    """)
                else:
                    st.error("❌ Unable to fetch weather data. Please check city name or API key.")
            except KeyError:
                st.error("🚨 OPENWEATHER_APIKEY is missing from secrets.toml")
            except Exception as e:
                st.error(f"🚨 Error fetching weather: {str(e)}")
        else:
            st.warning("Please enter a city name.")

    st.markdown('</div>')
else:
    pass  # Other sections remain unchanged

# Footer
st.markdown(f'<p style="text-align:center; font-size:14px;">{LANGUAGES[lang]["footer"]}</p>', unsafe_allow_html=True)

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
