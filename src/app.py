import streamlit as st
import json
import torch
from collections import Counter


# ===========================================#
#        Calling the transformer            #
# ===========================================#

from generate_name import NameGenerator

model = NameGenerator()

# ===========================================#
#              Streamlit Code               #
# ===========================================#
desc = "Uses a state of the art Transformer to generate a name for your project or enterprise!"

st.title("AI Namer")
st.write(desc)

# num_sentences = st.number_input(
#     "Number of Sentences", min_value=1, max_value=20, value=5
# )
user_input = st.text_input("Tell me what your project is all about!")


if st.button("Generate Text"):
    generated_text = model.generate(user_input)
    st.write(generated_text)
