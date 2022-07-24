# load libraries
import streamlit as st
from st_clickable_images import clickable_images
import os, glob, pathlib, random, pickle, time, requests, json, commons
import io
from io import StringIO, BytesIO
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sklearn
import cv2
from PIL import Image
from tqdm import tqdm
import tensorflow 
from tensorflow.python.client import device_lib
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input

# set remove keras messages
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

# set GPU for keras version
os.environ['CUDA_VISIBLE_DEVICES']='1'
physical_devices = tensorflow.config.list_physical_devices('GPU')
# print('physical_device:', physical_devices)
try:
    tensorflow.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
except RuntimeError as e:
    print(e)


# define class FeatureExtracor 
class FeatureExtractor:
    def __init__(self):
        # Use EfficientNetB1 as the architecture and ImageNet for the weight
        base_model = EfficientNetB1(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    def extract(self, img):
        # Resize the image
        img = img.resize((240, 240))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)


# define class ImageTagRetrieval
st.cache(persist=True)
class ImageTagRetrieval:
    def __init__(self):
        # prepare extracted features from features file
        features_file = "/nas3/epark/workspace/retreival/EfficientNetB2_features"
        with open(features_file, 'rb') as f:
            features = pickle.load(f)    
        # prepare img paths
        features_path = "/nas3/epark/workspace/retreival/feature_extraction/EfficientNetB1_new"
        imgs_dir = "/nas3/epark/workspace/retreival/unsplash_data_merge_new/"
        img_paths = []
        for feature_path in Path(features_path).glob('*.npy'):
            img_paths.append(Path(imgs_dir) / (feature_path.stem + '.jpg'))
              
        self.features, self.img_paths = features, img_paths
        self.fe = FeatureExtractor()

    def findtags(self, query_path):
        query_img = Image.open(BytesIO(requests.get(query_path).content))
        query_feature = self.fe.extract(query_img)
        dists = np.linalg.norm(self.features - query_feature, axis=1)
        ids = np.argsort(dists)[:15]
        samples = ids[:]
        samples_list = []
        samples_name = []
        frequency_tag_list = []
        for i in samples:
            p = self.img_paths[i]
            samples_list.append(p)
        
        for j in samples_list:
            name = j.parts[-1]
            samples_name.append(name)
        
        for item in samples_name:
            tag = item.split('_', maxsplit=1)
            tag = tag[0]
            frequency_tag_list.append(tag)
        
        count_tag, unique_tag = Counter(frequency_tag_list), set(frequency_tag_list)
        most_common = count_tag.most_common(3)
    
        if len(most_common) == 3:
            first_second = most_common[0][1] - most_common[1][1]
            second_third = most_common[1][1] - most_common[2][1]
            if first_second >= 2:
                final_tag = most_common[:1]
            elif first_second < 2 & second_third >= 2:
                final_tag = most_common[:2]
            else:
                final_tag = most_common
        
        elif len(most_common) == 2:
            first_second = most_common[0][1] - most_common[1][1]
            if first_second >= 2:
                final_tag = most_common[:1]
            else:
                final_tag = most_common[:2]
                
        else:
            final_tag = most_common[:]
            
        return final_tag


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


# streamlit pages2 main programming code
st.cache(persist=True)
def main():
    # show frontend title 
    st.title('Image to Music Retrieval')
    st.text("✔️ Please select an image! We recommend music that matches the selected image.")
    st.text("✔️ After selecting an image, please wait for a while until the next process.")

    # show imgs to be selected    
    selection = st.container()
    with selection:
        try:
            # show model load success message
            model_load_state = st.info('👉 Loooooooooooooooooooaaaaaaaaaaaaaaaaaaaading... 👀')
            model = ImageTagRetrieval()

            # url imgs
            imgs = [
                    'https://github.com/meungmeung/img2music_test/blob/main/prac_imgs/sport_action_adventure.jpg?raw=true',                    
                    'https://github.com/meungmeung/img2music_test/blob/main/prac_imgs/space.jpg?raw=true',
                    'https://github.com/meungmeung/img2music_test/blob/main/prac_imgs/sad_calm.jpg?raw=true',                  
                    'https://github.com/meungmeung/img2music_test/blob/main/prac_imgs/relaxing.jpg?raw=true',
                    'https://github.com/meungmeung/img2music_test/blob/main/prac_imgs/nature_calm_summer.jpg?raw=true'
                    'https://github.com/meungmeung/img2music_test/blob/main/prac_imgs/melodic_party.jpg?raw=true'
                    'https://github.com/meungmeung/img2music_test/blob/main/prac_imgs/love_relaxing_sad.jpg?raw=true'
                    ]

            # display images that can be clicked on using 'clickable_images' func
            clicked = clickable_images(paths=imgs, 
                                        titles=[f"Image {str(i)}" for i in range(1, len(imgs)+1)],
                                        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                                        img_style={"margin": "5px", "height": "200px"})  
            
            model_load_state.success('Show Up! Look Around Images!')
            final_tag = model.findtags(imgs[clicked])
            model_load_state.empty()
         
            # if some image is clicked,
            if clicked > -1:
                model_load_state.empty()
                model_load_state.info(f"**Retrieving Musics of Image {str(int(clicked)+1)}**")

                # show musics of selected image
                show_musics = st.container()
                with show_musics:
                    if len(final_tag) == 3:
                        # find musics using by tags
                        tags = [final_tag[0][0], final_tag[1][0], final_tag[2][0]]
                        f_tags = flatten_list(tags)     #f_tags = ['calm', 'advertising']
                        print('f_tags is', f_tags)
                        list_set = set(f_tags) # ['calm', 'advertising', 'calm] 일 경우, ['calm', 'advertising']
                        music_tag = list(list_set)
                        music_tag[:] = (value for value in music_tag if value != "-") # ['-']는 제외함
                
                        input_file = '/nas2/epark/mtg-jamendo-dataset/data/autotagging_moodtheme.tsv'
                        tracks, tags, extra = commons.read_file(input_file)
                        find_tag_list = []

                        for i in music_tag:
                            print('i', i)
                            p = tags['mood/theme'][i]
                            q = list(p)
                            find_tag_list.extend(q)
                            print('length: find_tag_list', find_tag_list)
                            
                        if len(find_tag_list) == 3:
                            a, b, c = find_tag_list
                            elements_in_all = list(set.intersection(*map(set, [a, b, c])))
                            elements_in_two = list(set.intersection(*map(set, [a, b])))
                            elements_in_two_2nd = list(set.intersection(*map(set, [b, c])))
                            elements_in_two_3rd = list(set.intersection(*map(set, [a, b])))
                            elements_in_one = a
                            
                            if len(elements_in_all) !=0 and len(elements_in_all) >= 5:
                                random_all = random.choices(elements_in_all, k=5)
                            elif len(elements_in_all) == 0 and len(elements_in_two) != 0  and len(elements_in_two) >= 5:
                                random_all = random.choices(elements_in_two, k=5)
                            elif len(elements_in_all) ==0 and len(elements_in_two) ==0 and len(elements_in_two_2nd) >= 5:
                                random_all = random.choices(elements_in_two_2nd, k=5)
                            elif len(elements_in_all) ==0 and len(elements_in_two) ==0 and len(elements_in_two_2nd) ==0 and len(elements_in_two_3rd) >=5:
                                random_all = random.choices(elements_in_two_3rd, k=5)
                            else:
                                random_all = random.choices(elements_in_one, k=5)

                            
                        elif len(find_tag_list) == 2:
                            a, b = find_tag_list
                            elements_in_all = list(set.intersection(*map(set, [a, b])))
                            elements_in_one = a
                            elements_in_one_2nd = b
                            
                            if len(elements_in_all) !=0 and len(elements_in_all) >= 5:
                                random_all = random.choices(elements_in_all, k=5)
                            elif len(elements_in_all) == 0 and len(elements_in_one) >= 5:
                                random_all = random.choices(elements_in_one, k=5)
                            else: 
                                random_all = random.choices(elements_in_one_2nd, k=5)

                            
                        else:
                            a = find_tag_list
                            elements_in_all = a
                            random_all = random.choices(elements_in_all, k=5)

                        # show up musics
                        with st.form('tags_3', clear_on_submit = True):
                            st.success(f"Success! Musics of Image {str(int(clicked)+1)} Are Below.")
                            st.subheader("Now, we recommend a music list that matches the image!")
                            st.text("🎧 Please enjoy the music and answer the questions below. 🎧")

                            audio_file1 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[0]) + '.mp3', 'rb')
                            audio_bytes1 = audio_file1.read()
                            st.audio(audio_bytes1, format='audio/ogg', start_time=0)

                            audio_file2 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[1]) + '.mp3', 'rb')
                            audio_bytes2 = audio_file2.read()
                            st.audio(audio_bytes2, format='audio/ogg', start_time=0)  

                            audio_file3 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[2]) + '.mp3', 'rb')
                            audio_bytes3 = audio_file3.read()
                            st.audio(audio_bytes3, format='audio/ogg', start_time=0)

                            audio_file4 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[3]) + '.mp3', 'rb')
                            audio_bytes4 = audio_file4.read()
                            st.audio(audio_bytes4, format='audio/ogg', start_time=0)

                            audio_file5 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[4]) + '.mp3', 'rb')
                            audio_bytes5 = audio_file5.read()
                            st.audio(audio_bytes5, format='audio/ogg', start_time=0)
                            
                            satis_result = st.slider('Do you satisfy with the recommended music?', min_value=0, max_value=100, value=50, step=1)

                            # save all submit
                            submitted = st.form_submit_button('SUBMIT')
                            print(submitted,"1")
                            if submitted:
                                model_load_state.info(f"**The Submission is Uploading.....**")
                                save_path = '/nas1/mingcha/img2music/retrieval/streamlit_v4_2/resultA.json'
                                results_A = {'Image': f"{str(int(clicked)+1)}", 'Tag1': final_tag[0][0], 'Tag2': final_tag[1][0], 'Tag3': final_tag[2][0], 'Music Satisfaction': satis_result}
                                if not os.path.exists(save_path):
                                    data = {}
                                    data['submits'] = []
                                    data['submits'].append(results_A)
                                    print("no exists", data)
                                    with open(save_path, 'w') as save_f:
                                        json.dump(data, save_f, ensure_ascii=False, indent=4) 
                                    model_load_state.info('**The submission is successfully uploaded.**')
                                
                                else:
                                    data = {}
                                    with open(save_path, "r") as json_file:
                                        data = json.load(json_file)
                                    data['submits'].append(results_A)
                                    print("exists, before", data)

                                    with open(save_path, "w") as save_f:
                                        json.dump(data, save_f, ensure_ascii=False, indent=4)
                                        print("exists, after", data)

                                    model_load_state.info('**The submission is successfully uploaded.**')



                    elif len(final_tag) == 2:
                        # find musics using by tags
                        tags = [final_tag[0][0], final_tag[1][0], '-']
                        f_tags = flatten_list(tags)     #f_tags = ['calm', 'advertising']
                        print('f_tags is', f_tags)
                        list_set = set(f_tags) # ['calm', 'advertising', 'calm] 일 경우, ['calm', 'advertising']
                        music_tag = list(list_set)
                        music_tag[:] = (value for value in music_tag if value != "-") # ['-']는 제외함
                
                        input_file = '/nas2/epark/mtg-jamendo-dataset/data/autotagging_moodtheme.tsv'
                        tracks, tags, extra = commons.read_file(input_file)
                        find_tag_list = []

                        for i in music_tag:
                            print('i', i)
                            p = tags['mood/theme'][i]
                            q = list(p)
                            find_tag_list.extend(q)
                            print('length: find_tag_list', find_tag_list)

                            
                        if len(find_tag_list) == 3:
                            a, b, c = find_tag_list
                            elements_in_all = list(set.intersection(*map(set, [a, b, c])))
                            elements_in_two = list(set.intersection(*map(set, [a, b])))
                            elements_in_two_2nd = list(set.intersection(*map(set, [b, c])))
                            elements_in_two_3rd = list(set.intersection(*map(set, [a, b])))
                            elements_in_one = a
                            
                            if len(elements_in_all) !=0 and len(elements_in_all) >= 5:
                                random_all = random.choices(elements_in_all, k=5)
                            elif len(elements_in_all) == 0 and len(elements_in_two) != 0  and len(elements_in_two) >= 5:
                                random_all = random.choices(elements_in_two, k=5)
                            elif len(elements_in_all) ==0 and len(elements_in_two) ==0 and len(elements_in_two_2nd) >= 5:
                                random_all = random.choices(elements_in_two_2nd, k=5)
                            elif len(elements_in_all) ==0 and len(elements_in_two) ==0 and len(elements_in_two_2nd) ==0 and len(elements_in_two_3rd) >=5:
                                random_all = random.choices(elements_in_two_3rd, k=5)
                            else:
                                random_all = random.choices(elements_in_one, k=5)

                            
                        elif len(find_tag_list) == 2:
                            a, b = find_tag_list
                            elements_in_all = list(set.intersection(*map(set, [a, b])))
                            elements_in_one = a
                            elements_in_one_2nd = b
                            
                            if len(elements_in_all) !=0 and len(elements_in_all) >= 5:
                                random_all = random.choices(elements_in_all, k=5)
                            elif len(elements_in_all) == 0 and len(elements_in_one) >= 5:
                                random_all = random.choices(elements_in_one, k=5)
                            else: 
                                random_all = random.choices(elements_in_one_2nd, k=5)

                            
                        else:
                            a = find_tag_list
                            elements_in_all = a
                            random_all = random.choices(elements_in_all, k=5)

                        
                        # show up musics
                        with st.form('tags_2', clear_on_submit=True):
                            st.success(f"Success! Musics of Image {str(int(clicked)+1)} Are Below.")
                            st.subheader("Now, we recommend a music list that matches the image!")
                            st.text("🎧 Please enjoy the music and answer the questions below. 🎧")

                            audio_file1 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[0]) + '.mp3', 'rb')
                            audio_bytes1 = audio_file1.read()
                            st.audio(audio_bytes1, format='audio/ogg', start_time=0)

                            audio_file2 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[1]) + '.mp3', 'rb')
                            audio_bytes2 = audio_file2.read()
                            st.audio(audio_bytes2, format='audio/ogg', start_time=0)  

                            audio_file3 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[2]) + '.mp3', 'rb')
                            audio_bytes3 = audio_file3.read()
                            st.audio(audio_bytes3, format='audio/ogg', start_time=0)

                            audio_file4 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[3]) + '.mp3', 'rb')
                            audio_bytes4 = audio_file4.read()
                            st.audio(audio_bytes4, format='audio/ogg', start_time=0)

                            audio_file5 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[4]) + '.mp3', 'rb')
                            audio_bytes5 = audio_file5.read()
                            st.audio(audio_bytes5, format='audio/ogg', start_time=0)
                            
                            satis_result = st.slider('Do you satisfy with the recommended music?', min_value=0, max_value=100, value=50, step=1)

                            # save all submit
                            submitted = st.form_submit_button('SUBMIT')
                            print(submitted,"1")
                            if submitted:
                                model_load_state.info(f"**The Submission is Uploading.....**")
                                save_path = '/nas1/mingcha/img2music/retrieval/streamlit_v4_2/resultA.json'
                                results_A = {'Image': f"{str(int(clicked)+1)}", 'Tag1': final_tag[0][0], 'Tag2': final_tag[1][0], 'Tag3': '-', 'Music Satisfaction': satis_result}
                                if not os.path.exists(save_path):
                                    data = {}
                                    data['submits'] = []
                                    data['submits'].append(results_A)
                                    print("no exists", data)
                                    with open(save_path, 'w') as save_f:
                                        json.dump(data, save_f, ensure_ascii=False, indent=4) 
                                    model_load_state.info('**The submission is successfully uploaded.**')
                                
                                else:
                                    data = {}
                                    with open(save_path, "r") as json_file:
                                        data = json.load(json_file)
                                    data['submits'].append(results_A)
                                    print("exists, before", data)

                                    with open(save_path, "w") as save_f:
                                        json.dump(data, save_f, ensure_ascii=False, indent=4)
                                        print("exists, after", data)

                                    model_load_state.info('**The submission is successfully uploaded.**')
                                


                    else:
                        # find musics using by tags
                        tags = [final_tag[0][0], '-', '-']
                        f_tags = flatten_list(tags)     #f_tags = ['calm', 'advertising']
                        print('f_tags is', f_tags)
                        list_set = set(f_tags) # ['calm', 'advertising', 'calm] 일 경우, ['calm', 'advertising']
                        music_tag = list(list_set)
                        music_tag[:] = (value for value in music_tag if value != "-") # ['-']는 제외함
                
                        input_file = '/nas2/epark/mtg-jamendo-dataset/data/autotagging_moodtheme.tsv'
                        tracks, tags, extra = commons.read_file(input_file)
                        find_tag_list = []

                        for i in music_tag:
                            print('i', i)
                            p = tags['mood/theme'][i]
                            q = list(p)
                            find_tag_list.extend(q)
                            print('length: find_tag_list', find_tag_list)

                            
                        if len(find_tag_list) == 3:
                            a, b, c = find_tag_list
                            elements_in_all = list(set.intersection(*map(set, [a, b, c])))
                            elements_in_two = list(set.intersection(*map(set, [a, b])))
                            elements_in_two_2nd = list(set.intersection(*map(set, [b, c])))
                            elements_in_two_3rd = list(set.intersection(*map(set, [a, b])))
                            elements_in_one = a
                            
                            if len(elements_in_all) !=0 and len(elements_in_all) >= 5:
                                random_all = random.choices(elements_in_all, k=5)
                            elif len(elements_in_all) == 0 and len(elements_in_two) != 0  and len(elements_in_two) >= 5:
                                random_all = random.choices(elements_in_two, k=5)
                            elif len(elements_in_all) ==0 and len(elements_in_two) ==0 and len(elements_in_two_2nd) >= 5:
                                random_all = random.choices(elements_in_two_2nd, k=5)
                            elif len(elements_in_all) ==0 and len(elements_in_two) ==0 and len(elements_in_two_2nd) ==0 and len(elements_in_two_3rd) >=5:
                                random_all = random.choices(elements_in_two_3rd, k=5)
                            else:
                                random_all = random.choices(elements_in_one, k=5)

                            
                        elif len(find_tag_list) == 2:
                            a, b = find_tag_list
                            elements_in_all = list(set.intersection(*map(set, [a, b])))
                            elements_in_one = a
                            elements_in_one_2nd = b
                            
                            if len(elements_in_all) !=0 and len(elements_in_all) >= 5:
                                random_all = random.choices(elements_in_all, k=5)
                            elif len(elements_in_all) == 0 and len(elements_in_one) >= 5:
                                random_all = random.choices(elements_in_one, k=5)
                            else: 
                                random_all = random.choices(elements_in_one_2nd, k=5)

                            
                        else:
                            a = find_tag_list
                            elements_in_all = a
                            random_all = random.choices(elements_in_all, k=5)

                        # show up musics
                        with st.form('tags_1', clear_on_submit=True):
                            st.success(f"Success! Musics of Image {str(int(clicked)+1)} Are Below.")
                            st.subheader("Now, we recommend a music list that matches the image!")
                            st.text("🎧 Please enjoy the music and answer the questions below. 🎧")

                            audio_file1 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[0]) + '.mp3', 'rb')
                            audio_bytes1 = audio_file1.read()
                            st.audio(audio_bytes1, format='audio/ogg', start_time=0)

                            audio_file2 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[1]) + '.mp3', 'rb')
                            audio_bytes2 = audio_file2.read()
                            st.audio(audio_bytes2, format='audio/ogg', start_time=0)  

                            audio_file3 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[2]) + '.mp3', 'rb')
                            audio_bytes3 = audio_file3.read()
                            st.audio(audio_bytes3, format='audio/ogg', start_time=0)

                            audio_file4 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[3]) + '.mp3', 'rb')
                            audio_bytes4 = audio_file4.read()
                            st.audio(audio_bytes4, format='audio/ogg', start_time=0)

                            audio_file5 = open('/nas3/epark/workspace/retreival/music_data/mp3/' + str(random_all[4]) + '.mp3', 'rb')
                            audio_bytes5 = audio_file5.read()
                            st.audio(audio_bytes5, format='audio/ogg', start_time=0)
                            
                            satis_result = st.slider('Do you satisfy with the recommended music?', min_value=0, max_value=100, value=50, step=1)

                            # save all submit
                            submitted = st.form_submit_button('SUBMIT')
                            print(submitted,"1")
                            if submitted:
                                model_load_state.info(f"**The Submission is Uploading.....**")
                                save_path = '/nas1/mingcha/img2music/retrieval/streamlit_v4_2/resultA.json'
                                results_A = {'Image': f"{str(int(clicked)+1)}", 'Tag1': final_tag[0][0], 'Tag2': '-', 'Tag3': '-', 'Music Satisfaction': satis_result}
                                if not os.path.exists(save_path):
                                    data = {}
                                    data['submits'] = []
                                    data['submits'].append(results_A)
                                    print("no exists", data)
                                    with open(save_path, 'w') as save_f:
                                        json.dump(data, save_f, ensure_ascii=False, indent=4) 
                                    model_load_state.info('**The submission is successfully uploaded.**')
                                
                                else:
                                    data = {}
                                    with open(save_path, "r") as json_file:
                                        data = json.load(json_file)
                                    data['submits'].append(results_A)
                                    print("exists, before", data)

                                    with open(save_path, "w") as save_f:
                                        json.dump(data, save_f, ensure_ascii=False, indent=4)
                                        print("exists, after", data)

                                    model_load_state.info('**The submission is successfully uploaded.**')
                                
                    
            # if not some image is clicked,
            else:
                model_load_state.info(f"**No Image Clicked. Click One, Please.**")
                
                
                
                


        except:
            message_container = st.empty() 
            message = message_container.write('👉 Please, wait. Loading... 👀')
            if message != '':
                time.sleep(23)
                message_container.empty()

            







if __name__ == '__main__':
    main()



