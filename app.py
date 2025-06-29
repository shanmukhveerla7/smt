import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from datetime import datetime
from fpdf import FPDF
import requests
import pandas as pd
import altair as alt

# Weather functions from uploaded file
def get_weather_data(city, weather_api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + weather_api_key + "&q=" + city
    response = requests.get(complete_url)
    return response.json()

def get_weekly_forecast(weather_api_key, lat, lon):
    base_url = "https://api.openweathermap.org/data/2.5/forecast?"
    complete_url = f"{base_url}lat={lat}&lon={lon}&appid={weather_api_key}"
    response = requests.get(complete_url)
    return response.json()

def display_weekly_forecast(data):
    try:
        st.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
        st.markdown("### 🗓️ Weekly Weather Forecast") 
        displayed_dates = set()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("", "Day")
        with c2:
            st.metric("", "Description")
        with c3:
            st.metric("", "Min Temp (°C)")
        with c4:
            st.metric("", "Max Temp (°C)")

        for day in data["list"]:
            date = datetime.fromtimestamp(day["dt"]).strftime("%A, %B %d")
            if date not in displayed_dates:
                displayed_dates.add(date)

                min_temp = day["main"]["temp_min"] - 273.15
                max_temp = day["main"]["temp_max"] - 273.15
                description = day["weather"][0]["description"]

                with c1:
                    st.write(f"**{date.split(',')[0]}**")
                with c2:
                    st.write(description.capitalize())
                with c3:
                    st.write(f"{min_temp:.1f}°C")
                with c4:
                    st.write(f"{max_temp:.1f}°C")

    except Exception as e:
        st.error("Error displaying forecast: " + str(e))

def generate_forecast_summary1(forecast_data, openai_api_key):
    st.warning("OpenAI integration temporarily disabled. Using local summary.")
    # Simulated fallback using Watsonx or simple string
    try:
        text_block = "Weekly Forecast Summary:\n"
        for entry in forecast_data["list"][:8]:  # First 24 hours
            dt_txt = entry["dt_txt"]
            desc = entry["weather"][0]["description"]
            temp = entry["main"]["temp"] - 273.15
            text_block += f"{dt_txt}: {desc}, {temp:.1f}°C\n"

        return text_block
    except Exception as e:
        st.error("Error generating summary: " + str(e))
        return None

def plot_forecast_chart(forecast_data):
    daily_data = []
    seen_dates = set()

    for entry in forecast_data["list"]:
        date_str = entry["dt_txt"].split(" ")[0]
        if date_str not in seen_dates:
            seen_dates.add(date_str)
            daily_data.append({
                "Date": date_str,
                "Min Temp (°C)": entry["main"]["temp_min"] - 273.15,
                "Max Temp (°C)": entry["main"]["temp_max"] - 273.15,
                "Humidity (%)": entry["main"]["humidity"],
                "Wind Speed (m/s)": entry["wind"]["speed"]
            })

    df = pd.DataFrame(daily_data)

    st.subheader("📉 Temperature Forecast")
    temp_chart = (
        alt.Chart(df)
        .transform_fold(["Min Temp (°C)", "Max Temp (°C)"], as_=["Type", "Temperature"])
        .mark_line(point=True)
        .encode(x="Date:T", y="Temperature:Q", color="Type:N")
        .properties(width=700)
    )
    st.altair_chart(temp_chart)

    st.subheader("💧 Humidity & Wind Speed Forecast")
    hum_wind_chart = (
        alt.Chart(df)
        .transform_fold(["Humidity (%)", "Wind Speed (m/s)"], as_=["Type", "Value"])
        .mark_line(point=True)
        .encode(x="Date:T", y="Value:Q", color="Type:N")
        .properties(width=700)
    )
    st.altair_chart(hum_wind_chart)

def get_air_pollution_data(lat, lon, weather_api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={weather_api_key}"
    response = requests.get(url)
    return response.json()

def display_air_pollution(data):
    st.markdown("### 🌫️ Air Quality Index (AQI)")
    try:
        aqi = data["list"][0]["main"]["aqi"]
        aqi_text = {
            1: "Good 😊",
            2: "Fair 🙂",
            3: "Moderate 😐",
            4: "Poor 🥶",
            5: "Very Poor 🤢"
        }
        st.metric("Air Quality", f"{aqi} - {aqi_text.get(aqi, 'Unknown')}")

        st.markdown("#### Pollutants (μg/m³):")
        pollutants = data["list"][0]["components"]
        for key, value in pollutants.items():
            st.write(f"**{key.upper()}**: {value}")
    except Exception as e:
        st.error("Error displaying air pollution data: " + str(e))


# Language translations
LANGUAGES = {
    "en": {
        "title": "🌆 Smart City Assistant",
        "subtitle": "Ask about traffic, energy, environment, and infrastructure.",
        "chat": "🤖 AI Chatbot",
        "traffic": "🚦 Traffic Monitor",
        "energy": "⚡ Energy Tracker",
        "environment": "🌍 Environmental Insights",
        "weather": "🌦️ Weather Forecast",
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
        "weather": "🌦️ Pronóstico del Tiempo",
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
        "weather": "🌦️ Météo",
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

# Custom CSS - Unique card styles per dashboard
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
            color: #343a40;
        }

        .main {
            padding: 30px;
            max-width: 1200px;
            margin: auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.07);
        }

        h1, h2, h3 {
            color: #2c3e50;
            font-size: 20px;
            font-weight: 600;
        }

        label {
            font-weight: 500;
            color: #495057;
            font-size: 14px;
        }

        input, select, textarea {
            border-radius: 6px;
            border: 1px solid #ced4da;
            padding: 10px;
            width: 100%;
        }

        button {
            background-color: #007AFF;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 6px;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3;
        }

        /* Navigation Bar */
        .navbar {
            display: flex;
            justify-content: center;
            gap: 20px;
            padding: 10px 0;
            background: #ffffff;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        .nav-button {
            background-color: #ffffff;
            color: #2c3e50;
            border: 1px solid #ccc;
            width: 140px;
            height: 40px;
            font-size: 14px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .nav-button:hover {
            background-color: #def8ff;
            color: #007AFF;
            border-color: #bbb;
        }

        .nav-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Dashboard Card Styles */
        .card-chat {
            background-color: #f3f9fb;
            padding: 22px;
            margin: 15px 0;
            border-left: 6px solid #007AFF;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card-traffic {
            background-color: #fff7f6;
            padding: 22px;
            margin: 15px 0;
            border-left: 6px solid #E63946;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card-energy {
            background-color: #f2fbf6;
            padding: 22px;
            margin: 15px 0;
            border-left: 6px solid #2A9D8F;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card-environment {
            background-color: #f0f7ff;
            padding: 22px;
            margin: 15px 0;
            border-left: 6px solid #264DE4;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card-weather {
            background-color: #fffbe6;
            padding: 22px;
            margin: 15px 0;
            border-left: 6px solid #F4A261;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card-reports {
            background-color: #f5f5f5;
            padding: 22px;
            margin: 15px 0;
            border-left: 6px solid #6A7581;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card-settings {
            background-color: #f5f5f5;
            padding: 22px;
            margin: 15px 0;
            border-left: 6px solid #999;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card-weather:hover, .card-traffic:hover, .card-energy:hover, .card-environment:hover, .card-chat:hover, .card-reports:hover, .card-settings:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .user-bubble, .bot-bubble {
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 12px;
            max-width: 70%;
            font-size: 14px;
        }

        .user-bubble {
            background-color: #007AFF;
            color: white;
            align-self: flex-end;
        }

        .bot-bubble {
            background-color: #ecf0f1;
            color: black;
            align-self: flex-start;
        }

        .footer {
            text-align: center;
            font-size: 14px;
            color: #6c757d;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #e9ecef;
        }
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
        if st.button(LANGUAGES[lang]["chat"], key="btn_chat", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "chat"
    with col2:
        if st.button(LANGUAGES[lang]["traffic"], key="btn_traffic", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "traffic"
    with col3:
        if st.button(LANGUAGES[lang]["energy"], key="btn_energy", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "energy"
    with col4:
        if st.button(LANGUAGES[lang]["environment"], key="btn_environment", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "environment"
    with col5:
        if st.button(LANGUAGES[lang]["weather"], key="btn_weather", use_container_width=True, disabled=not st.session_state.profile_complete):
            st.session_state.current_section = "weather"
    with col6:
        if st.button("🧾", key="btn_profile", use_container_width=True):
            st.session_state.current_section = "profile"
    st.markdown('Unlock the full potential of our assistant to address your questions efficiently!!', unsafe_allow_html=True)

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
    st.rerun()

def reset_profile():
    st.session_state.profile_complete = False
    st.session_state.profile_data = {}
    st.session_state.messages = []
    st.session_state.city_data = {}
    st.rerun()

# ------------------------------ SETTINGS ------------------------------
if st.session_state.current_section == "settings":
    st.markdown('<div class="card-settings">', unsafe_allow_html=True)
    st.markdown(f'<h2>⚙️ {LANGUAGES[lang]["settings"]}</h2>', unsafe_allow_html=True)
    language = st.selectbox("Language", options=["en", "es", "fr"], format_func=lambda x: {"en": "English", "es": "Español", "fr": "Français"}[x])
    theme = st.selectbox("Theme", ["Light"])
    font_size = st.slider("Font Size", 12, 24)
    if st.button(LANGUAGES[lang]["save_profile"]):
        st.session_state.language = language
        st.success("Preferences updated!")
    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

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
    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

# If profile not completed, show message only
elif not st.session_state.profile_complete:
    st.info("ℹ️ Please complete your profile before continuing.", icon="🪪")
    if st.button("Go to Profile"):
        st.session_state.current_section = "profile"
    st.stop()

# ------------------------------ CHATBOT ------------------------------
elif st.session_state.current_section == "chat":
    st.markdown('<div class="card-chat">', unsafe_allow_html=True)
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
    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

# ------------------------------ TRAFFIC MONITOR ------------------------------
elif st.session_state.current_section == "traffic":
    st.markdown('<div class="card-traffic">', unsafe_allow_html=True)
    st.markdown('<h2>🚦 Traffic Monitor</h2>', unsafe_allow_html=True)
    query = st.text_area("Describe your traffic-related issue or question:")
    if st.button("Get Advice"):
        llm = get_llm("traffic")
        res = llm.invoke(query)
        st.markdown(f'<div class="bot-bubble">{res}</div>', unsafe_allow_html=True)
    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

# ------------------------------ ENERGY TRACKER ------------------------------
elif st.session_state.current_section == "energy":
    st.markdown('<div class="card-energy">', unsafe_allow_html=True)
    st.markdown('<h2>⚡ Energy Tracker</h2>', unsafe_allow_html=True)
    query = st.text_input("Ask about power usage or grid issues:")
    if st.button("Get Suggestions"):
        llm = get_llm("energy")
        res = llm.invoke(query)
        st.markdown(f'<div class="bot-bubble">{res}</div>', unsafe_allow_html=True)
    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

# ------------------------------ ENVIRONMENT ANALYSIS ------------------------------
elif st.session_state.current_section == "environment":
    st.markdown('<div class="card-environment">', unsafe_allow_html=True)
    st.markdown('<h2>🌍 Environmental Insights</h2>', unsafe_allow_html=True)
    query = st.text_area("Ask about pollution, air quality, or sustainability:")
    if st.button("Get Insight"):
        llm = get_llm("environment")
        res = llm.invoke(query)
        st.markdown(f'<div class="bot-bubble">{res}</div>', unsafe_allow_html=True)
    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

# ------------------------------ WEATHER DASHBOARD ------------------------------
elif st.session_state.current_section == "weather":
    st.markdown('<div class="card-weather">', unsafe_allow_html=True)
    st.markdown('<h2>🌦️ Weather Forecast</h2>', unsafe_allow_html=True)

    city = st.text_input("Enter City Name")
    try:
        weather_api_key = st.secrets["OPENWEATHER_APIKEY"]
    except KeyError:
        st.error("🚨 OpenWeatherMap API key missing in secrets.toml")
        st.stop()

    if st.button("Get Current Weather"):
        if city:
            try:
                data = get_weather_data(city, weather_api_key)
                if data.get("cod") != 200:
                    st.error("❌ Unable to fetch weather data. Check city name or API key.")
                else:
                    current_temp = data["main"]["temp"] - 273.15
                    feels_like = data["main"]["feels_like"] - 273.15
                    humidity = data["main"]["humidity"]
                    wind_speed = data["wind"]["speed"]
                    desc = data["weather"][0]["description"]

                    st.markdown(f"""
                        **City:** {data['name']}  
                        **Temperature:** {current_temp:.1f}°C  
                        **Feels Like:** {feels_like:.1f}°C  
                        **Humidity:** {humidity}%  
                        **Wind Speed:** {wind_speed} m/s  
                        **Description:** {desc.capitalize()}
                    """)
            except Exception as e:
                st.error(f"🚨 Error fetching weather: {str(e)}")
        else:
            st.warning("Please enter a city name.")

    if st.button("Get Weekly Forecast"):
        if city:
            try:
                data = get_weather_data(city, weather_api_key)
                if data.get("cod") != 200:
                    st.error("❌ Unable to fetch weather data. Check city name or API key.")
                else:
                    lat = data["coord"]["lat"]
                    lon = data["coord"]["lon"]
                    forecast_data = get_weekly_forecast(weather_api_key, lat, lon)
                    summary = generate_forecast_summary1(forecast_data, st.secrets["WATSONX_APIKEY"])
                    st.markdown("🧠 **AI Summary:**")
                    st.markdown(summary)
                    plot_forecast_chart(forecast_data)
            except Exception as e:
                st.error(f"🚨 Error fetching weekly forecast: {str(e)}")
        else:
            st.warning("Please enter a city name.")

    if st.button("Get Air Quality"):
        if city:
            try:
                data = get_weather_data(city, weather_api_key)
                lat = data["coord"]["lat"]
                lon = data["coord"]["lon"]
                aqi_data = get_air_pollution_data(lat, lon, weather_api_key)
                display_air_pollution(aqi_data)
            except Exception as e:
                st.error(f"🚨 Error fetching air quality: {str(e)}")
        else:
            st.warning("Please enter a city name.")

    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

# ------------------------------ PROGRESS REPORTS ------------------------------
elif st.session_state.current_section == "reports":
    st.markdown('<div class="card-reports">', unsafe_allow_html=True)
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
        st.markdown(f'<div class="bot-bubble">{summary}</div>', unsafe_allow_html=True)

    if st.session_state.profile_complete and st.session_state.city_data:
        st.download_button(
            label=LANGUAGES[lang]["export_pdf"],
            data=export_city_report(),
            file_name="city_report.pdf",
            mime="application/pdf"
        )
    st.markdown('Grateful for your time—our assistant is here to help anytime you need!!')

# Footer
st.markdown(f'<p class="footer">{LANGUAGES[lang]["footer"]}</p>', unsafe_allow_html=True)

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
