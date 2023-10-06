import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import math 

# Sayfa Ayarları
st.set_page_config(
    page_title="Collaborative Based Product Recommendation Systems",
    page_icon="https://cdn.icon-icons.com/icons2/1195/PNG/512/1490889698-amazon_82521.png",
    menu_items={ 
        "About": "For More Information\n" + "https://github.com/cinarolog"
    }
)


st.title("Collaborative Based Product Recommendation System")
st.image("images/collaborative.jpeg")

collab_df = pd.read_csv("collab_df.csv")


def collab_recommend_products(df, user_id_encoded):
    # Use TfidfVectorizer to transform the product descriptions into numerical feature vectors
    tfidf = TfidfVectorizer(stop_words='english')
    df['review_content'] = df['review_content'].fillna('')  # fill NaN values with empty string
    tfidf_matrix = tfidf.fit_transform(df['review_content'])

    # Get the purchase history for the user
    user_history = df[df['user_id_encoded'] == user_id_encoded]

    # Use cosine_similarity to calculate the similarity between each pair of product descriptions
    # only for the products that the user has already purchased
    indices = user_history.index.tolist()

    if indices:
        # Create a new similarity matrix with only the rows and columns for the purchased products
        cosine_sim_user = cosine_similarity(tfidf_matrix[indices], tfidf_matrix)

        # Create a pandas Series with product indices as the index and product names as the values
        products = df.iloc[indices]['product_name']
        indices = pd.Series(products.index, index=products)

        # Get the indices and similarity scores of products similar to the ones the user has already purchased
        similarity_scores = list(enumerate(cosine_sim_user[-1]))
        similarity_scores = [(i, score) for (i, score) in similarity_scores if i not in indices]

        # Sort the similarity scores in descending order
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top 5 most similar products
        top_products = [i[0] for i in similarity_scores[1:6]]
        
         # Get the img_link of the top 5 most similar products
        img_link = df.iloc[top_products]['img_link'].tolist()

         # Get the img_link of the top 5 most similar products
        product_link = df.iloc[top_products]['product_link'].tolist()


        # Get the names of the top 5 most similar products
        recommended_products = df.iloc[top_products]['product_name'].tolist()

        # Get the reasons for the recommendation
        score = [similarity_scores[i][1] for i in range(5)]

        # Create a DataFrame with the results
        results_df = pd.DataFrame({'Id Encoded': [user_id_encoded] * 5,
                                   'recommended product': recommended_products,
                                   'img_link': img_link ,
                                   'product_link': product_link ,
                                   'score recommendation': score})


        return results_df

    else:
        print("No purchase history found.")
        return None


user_id_encoded= st.sidebar.number_input("Number of Usert_id", min_value=1, format="%d")


df_products=collab_recommend_products(collab_df,user_id_encoded)

if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

   
    st.table(collab_recommend_products(collab_df,user_id_encoded))

else:
    st.markdown("Please click the *Submit Button*!")

choose_product_with_num= st.sidebar.number_input("Choose Product index", min_value=1, format="%d")

if st.button("Find Product"):

    # Info mesajı oluşturma
    st.info("You can find the products link&img and more below...")
    st.markdown(df_products.product_link[choose_product_with_num])
    st.image(df_products.img_link[choose_product_with_num])
    
else:
    st.markdown("Please click the *Submit Button* After that choose prodcut num and click Find Product")







