import asyncio
from contextlib import contextmanager
from sklearn import datasets
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf

@contextmanager
def setup_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)

# Use the context manager to create an event loop
with setup_event_loop() as loop:
    import sketch

st.cache_data()
def get_iris_data():
    data = datasets.load_iris()
    iris_data  = pd.DataFrame(data=data.data, columns=data.feature_names)
    iris_data['species'] = pd.Categorical.from_codes(data.target, data.target_names)
    return iris_data

st.set_page_config('Plotting Iris Data', layout = 'wide')
iris_data = get_iris_data()
if not st.sidebar.checkbox('Hide Raw Data'):
    st.dataframe(iris_data,use_container_width=True)

question = st.sidebar.text_input('Type your question here!')
if 'recording_ind' not in st.session_state:
        st.session_state['recording_ind'] = False

if 'prev_recording_ind' not in st.session_state:
    st.session_state['prev_recording_ind'] = None



# Create a placeholder for displaying the transcribed text
text_placeholder = st.empty()

# Create a recognizer object for the SpeechRecognition library
r = sr.Recognizer()

text = None
# Record audio when the "Start Recording" button is pressed
if st.sidebar.button('Speak Now'):
    with st.spinner('Recording for 5 seconds'):
        duration = 5  # seconds
        filename = 'temp.wav'
        samplerate = 44100
        channels = 1
        myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        sd.wait()  # Wait for recording to complete

        #st.write("Stopped recording.")
        # Save the recorded audio to a WAV file
        sf.write(filename, myrecording, samplerate)

        # Load the audio file into memory
        with sr.AudioFile(filename) as source:
            audio = r.record(source)

        # Convert speech to text
        text = r.recognize_google(audio)

    # Display the transcribed text
    #st.write("Transcribed text:")
    st.sidebar.write(text)

if question:
    st.info(iris_data.sketch.ask(question,call_display=False))

if text:
    st.info(iris_data.sketch.ask(text,call_display=False))

@st.cache_data
def plot_graphs(iris_data):
    col1, col2 = st.columns(2)
    with col1:
        st.write(' ')
        for _i in range(0,1 ):
            st.write(' ')
        fig = plt.figure()
        fig_histplot = sns.histplot(iris_data['petal length (cm)'])
        st.pyplot(fig)

        st.write(' ')
        fig = plt.figure()
        sns.kdeplot(data=iris_data['petal length (cm)'],fill=True)
        st.pyplot(fig)

        fig = sns.jointplot(x=iris_data['petal length (cm)'], y=iris_data['sepal width (cm)'], kind="kde")
        st.pyplot(fig)

    with col2:
        fig = plt.figure()
        sns.histplot(data=iris_data, x='petal length (cm)', hue='species')
        plt.title("Histogram of Petal Lengths, by Species")
        st.pyplot(fig)

        fig = plt.figure()
        sns.kdeplot(data=iris_data, x='petal length (cm)', hue='species',fill=True)
        plt.title("Distribution of Petal Lengths, by Species")
        st.pyplot(fig)

        for _i in range(0,6):
            st.write(' ')
        fig = plt.figure()
        sns.regplot(x=iris_data['sepal length (cm)'], y=iris_data['petal width (cm)'], data = iris_data)
        st.pyplot(fig)

plot_graphs(iris_data)