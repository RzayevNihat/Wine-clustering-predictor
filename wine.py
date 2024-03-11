# Lazım olan kitabxanalar
import streamlit as st
import pandas as pd
import warnings
import pickle
import time
from PIL import Image

# Potensial xəbərdarlıqların filterlənməsi
warnings.filterwarnings(action = 'ignore')

# Datasetin yüklənməsi
df = pd.read_csv(filepath_or_buffer = 'wine-clustering.csv')

# Sütun adlarının kiçildilməsi və potensial boşluqların silinməsi
df.columns = df.columns.str.lower().str.strip()

# Modelin yülənməsi
with open(file= 'kmeans.pickle',mode='rb') as pickled_model:
    model=pickle.load(file=pickled_model)
# Şəkilin yüklənməsi
wine_image = Image.open('wine-clustering.jpeg')
    
# Əsas səhifənin yaradılması
interface = st.container()

# Əsas səhifəyə elementlərin daxil edilməsi
with interface:
    # Səhifənin adının göstərilməsi (səhifə adı --> Wine Quality Clustering)
    st.title(body='Wine Quality Clustering')
    
    # Şəkilin göstərilməsi
    st.image(image=wine_image)
    
    # Başlığın göstərilməsi (başlıq adı --> Project Description)
    st.header(body='Project Description')
    # Proyekt haqqında informasiyanın verilməsi
    st.markdown(body = f"""This is a machine learning project in which wines are clustered based on their quality. 
    KMeans algoritm was used to build the model with **{df.shape[1]}** features. Principal Component Analysis was 
    used to reduce dimensionality whereas the number of clusters was identified using Elbow method.""")
    
    # Kiçik başlığın göstərilməsi (kiçik başlıq adı --> Input Features)
    st.subheader(body='Input Features')    
    # Düz xəttin çəkilməsi
    st.markdown(body = '***')
    
    # Asılı olmayan dəyişənlərin yaradılması (Bütün asılı olmayan dəyişənləri st.slider() metodu ilə yarat)
    alcohol = st.slider(label='Alcohol', min_value=float(df.alcohol.min()), max_value=float(df.alcohol.max()), value=float(df.alcohol.mean()))
    malic_acid = st.slider(label='Malic Acid', min_value=float(df.malic_acid.min()), max_value=float(df.malic_acid.max()), value=float(df.malic_acid.mean()))
    ash = st.slider(label='Ash', min_value=float(df.ash.min()), max_value=float(df.ash.max()), value=float(df.ash.mean()))
    ash_alcanity = st.slider(label='Ash Alcanity', min_value=float(df.ash_alcanity.min()), max_value=float(df.ash_alcanity.max()), value=float(df.ash_alcanity.mean()))
    magnesium = st.slider(label='Magnesium', min_value=int(df.magnesium.min()), max_value=int(df.magnesium.max()), value=int(df.magnesium.mean()))
    total_phenols = st.slider(label='Total Phenols', min_value=float(df.total_phenols.min()), max_value=float(df.total_phenols.max()), value=float(df.total_phenols.mean()))
    flavanoids = st.slider(label='Flavanoids', min_value=float(df.flavanoids.min()), max_value=float(df.flavanoids.max()), value=float(df.flavanoids.mean()))
    nonflavanoid_phenols = st.slider(label='Nonflavanoid Phenols', min_value=float(df.nonflavanoid_phenols.min()), max_value=float(df.nonflavanoid_phenols.max()), value=float(df.nonflavanoid_phenols.mean()))
    proanthocyanins = st.slider(label='Proanthocyanins', min_value=float(df.proanthocyanins.min()), max_value=float(df.proanthocyanins.max()), value=float(df.proanthocyanins.mean()))
    color_intensity = st.slider(label='Color Intensity', min_value=float(df.color_intensity.min()), max_value=float(df.color_intensity.max()), value=float(df.color_intensity.mean()))
    hue = st.slider(label='Hue', min_value=float(df.hue.min()), max_value=float(df.hue.max()), value=float(df.hue.mean()))
    od280 = st.slider(label='OD280', min_value=float(df.od280.min()), max_value=float(df.od280.max()), value=float(df.od280.mean()))
    proline = st.slider(label='Proline', min_value=int(df.proline.min()), max_value=int(df.proline.max()), value=int(df.proline.mean()))

    
    # Düz xəttin çəkilməsi
    st.markdown(body = '***')
    
    # Kiçik başlığın göstərilməsi (kiçik başlıq adı --> Making Predictions)
    st.subheader(body='Making Predictions')
    
    # Lügət data strukturunun yaradılması
    data_dicionary = {'alcohol':alcohol,
                       'malic_acid':malic_acid,
                       'ash':ash,
                       'ash_alcanity':ash_alcanity,
                       'magnesium':magnesium,
                       'total_phenols':total_phenols,
                       'flavanoids':flavanoids,
                       'nonflavanoid_phenols':nonflavanoid_phenols,
                       'proanthocyanins':proanthocyanins,
                       'color_intensity':color_intensity,
                       'hue':hue,
                       'od280':od280,
                       'proline':proline}
    
    # Lügət data strukturunun DataFrame data strukturuna çevirilməsi
    input_features = pd.DataFrame(data=data_dicionary,index=[0])
    st.subheader(body=f'{input_features.shape}')

    # Proqnoz adlarının yaradılması
    cluster_labels = {0:'first', 1:'second', 2:'third'}
    
    # Predict adında düymənin yaradılması
    if st.button('Predict'):
        # Döngünün yaradılması
        with st.spinner(text='Sending input features to model...'):
            # İki saniyəlik pauzanın yaradılması
            time.sleep(2)
        
        # Klasterin model tərəfindən proqnozlaşdırılması
        predicted_cluster = model.fit_predict(X = df).ravel()[1]
        
        # Klasterin adının əldə olunması
        cluster_label = cluster_labels.get(predicted_cluster)
        
        # Proqnozun verilməsi ilə bağlı mesajın göstərilməsi
        st.success('Prediction is ready')
        
        # Bir saniyəlik pauzanın yaradılması
        time.sleep(1)
        
        # Proqnozun istifadəçiyə göstərilməsi
        st.markdown(f'Model output: Wine belongs to the **{cluster_label}** cluster')