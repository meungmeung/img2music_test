import streamlit as st
import pandas as pd
import os, json

st.subheader("please respond to the questionnaire consisting of 4 sections.")
st.subheader("💪 You are almost there! 💪")

# @st.cache(persist=True)
survey = st.container()


with survey:
    with st.form('Survey'):
        st.text("############## First Section ##############")
        gender = st.radio(
            "What's your gender?",
            ('Male', 'Female', 'Intersex'))

        age = st.radio(
            "What's your age group?",
            ('10s', '20s', '30s', '40s', '50s', '60s', 'Above 60s'))

        education = st.radio(
            "What's the highest level of education that you have completed?",
            ('Primary/Elementary education not completed', 'Primary/Elementary education', 'Secondary education','Further education (Bachelor degree, diploma', 'Higher education (Masters, Doctorate)'))


        ethnicity = st.radio(
            "What's your Ethnicity (or Race)?",
            ('Prefer not to disclose', 'American Indigenous (Alaskan Native / Native American)', 'Asian', 'Black', 'Latinx / Hispanic', 'Middle Eastern / North African', 'Pacific Islander', 'White / Caucasian', 'Multi Race / Ethnicity'))
        
        st.text("############## Second Section ##############")
        satisfaction1 = st.radio(
            "I'm very satisfied with this image-music retrievel system.",
            ('Very satisfied', 'Satisfied', 'Neutral', 'Unsatisfied', 'Very unsatisfied'))

        satisfaction2 = st.radio(
            "This image-music retrieval system meet my expectation.",
            ('Strongly agree', 'Agree', 'Neutral', 'Disagree', 'Strongly disagree'))

        satisfaction3 = st.radio(
            "This image-music retrieval system work the way I want it to work.",
            ('Strongly agree', 'Agree', 'Neutral', 'Disagree', 'Strongly disagree'))

        st.text("############## Third Section ##############")
        ItU1 = st.radio(
            "I intend to use this image-music retrieval system in the future.",
            ('Strongly agree', 'Agree', 'Neutral', 'Disagree', 'Strongly disagree'))

        ItU2 = st.radio(
            "I plan to use this image-music retrieval system when it starts to service.",
            ('Strongly agree', 'Agree', 'Neutral', 'Disagree', 'Strongly disagree'))

        ItU3 = st.radio(
            "I would recommend this image-music retrieval system to others.",
            ('Strongly agree', 'Agree', 'Neutral', 'Disagree', 'Strongly disagree'))

        # save all submit
        submitted = st.form_submit_button('SUBMIT')
        print(submitted,"4")
        if submitted:
            save_path = '/nas1/mingcha/img2music/retrieval/streamlit_v4_2/resultA.json'
            results_A = {'gender': gender, 'age': age, 'education': education, 'ethnicity': ethnicity, 'satisfaction1': satisfaction1, 'satisfaction2': satisfaction2, 'satisfaction3': satisfaction3, 'ItU1': ItU1, 'ItU2': ItU2, 'ItU3': ItU3}
            data = {}
            with open(save_path, "r") as json_file:
                data = json.load(json_file)
            data['submits'].append(results_A)

            with open(save_path, "w") as save_f:
                json.dump(data, save_f, ensure_ascii=False, indent=4)
                print("exists, after", data)
            st.info('**The submission is successfully uploaded.**')

# if st.button("SUBMIT"):
#     get_data().append({"sex": sex, "age": age})

# st.write(pd.DataFrame(get_data()))



# if 'num' not in st.session_state:
#     st.session_state.num = 0


# choices1 = ['Male', 'Female', 'Intersex']
# choices2 = ['10s', '20s', '30s', '40s', '50s', '60s', 'Above 60s']
# choices3 = ['Primary/Elementary education not completed', 'Primary/Elementary education', 'Secondary education','Further education (Bachelor degree, diploma', 'Higher education (Masters, Doctorate)']
# choices4 = ['Prefer not to disclose', 'American Indigenous (Alaskan Native / Native American)', 'Asian', 'Black', 'Latinx / Hispanic', 'Middle Eastern / North African', 'Pacific Islander', 'White / Caucasian', 'Multi Race / Ethnicity']
# choices5 = ['Very satisfied', 'Satisfied', 'Neutral', 'Unsatisfied', 'Very unsatisfied']
# choices6 = ['Strongly agree', 'Agree', 'Neutral', 'Disagree', 'Strongly disagree']

# qs1 = [("What's your gender?", choices1),
#     ("I'm satisfied with Image-Music retrieval results", choices5),
#     ('Image-Music retrieval system work the way I want them to work', choices6)]
# qs2 = [("What's your age group?", choices2),
#     ('Image-Music retrieval system is pleased to use', choices6),
#     ('I intend to use Image-Music retrieval system in the future', choices6)]
# qs3 = [("What's the highest level of education that you have completed?", choices3),
#     ('Image-Music retrieval system is pleased to use', choices6),
#     ('I intend to use Image-Music retrieval system in the future', choices6)]
# qs4 = [("What's your Ethnicity (or Race)?", choices4),
#     ('Image-Music retrieval system is pleased to use', choices6),
#     ('I intend to use Image-Music retrieval system in the future', choices6)]

# def main():
#     for _, _, _, _ in zip(qs1, qs2, qs3, qs4): 
#         placeholder = st.empty()
#         num = st.session_state.num
#         with placeholder.form(key=str(num)):
#             st.radio(qs1[num][0], key=num+1, options=qs1[num][1])
#             st.radio(qs2[num][0], key=num+1, options=qs2[num][1])          
#             st.radio(qs3[num][0], key=num+1, options=qs3[num][1])
#             st.radio(qs4[num][0], key=num+1, options=qs4[num][1])       

#             if st.form_submit_button():
#                 st.session_state.num += 1
#                 if st.session_state.num >= 5:
#                     st.session_state.num = 0 
#                 placeholder.empty()
#             else:
#                 st.stop()

# main()