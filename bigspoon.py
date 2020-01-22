import streamlit as st
st.write("First App Evaaaa")

import requests

st.subheader('existing letters')
letters = requests.get('http://127.0.0.1:8000/letters').json()
for letter in letters:
    st.write(letter)

st.subheader('add new letters')
new_candidates = []
for new_letter in ['d', 'e', 'f']:
    st.write(f'add {new_letter}')
    add_letter = st.checkbox(f'add {new_letter}')
    if add_letter:
        new_candidates.append(new_letter)

submit = st.button('submit new letters')
if submit:
    requests.post('http://127.0.0.1:8000/letters', json=new_candidates)