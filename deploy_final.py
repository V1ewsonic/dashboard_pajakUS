# ------- Library ----------- #
import streamlit as st
import time
import pandas as pd
import plotly.express as ps
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import altair as alt
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import warnings
warnings.filterwarnings('ignore')




# ------- FUNCTION ------------ #
# Page configuration
st.set_page_config(
    page_title="Wajib Pajak US",
    page_icon="	:iphone:",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("default")


def data_load(datas):
  import pandas as pd
  data = pd.read_csv(datas)
  return data

# Fungsi pengecekkan apakah format yang diupload CSV
def is_csv(filename):
    return filename.lower().endswith('.csv')

# Inisialisasi session state untuk button
if 'button1_clicked' not in st.session_state:
    st.session_state.button1_clicked = False

if 'button2_clicked' not in st.session_state:
    st.session_state.button2_clicked = False

if 'button3_clicked' not in st.session_state:
    st.session_state.button3_clicked = False

# Fungsi untuk mengubah status button
def click_button1():
    st.session_state.button1_clicked = True

def click_button2():
    st.session_state.button2_clicked = True

def click_button3():
    st.session_state.button3_clicked = True

def highlight_column(s):
    return['background-color : lightyellow' if s.name == 'cluster' else '' for _ in s]


# FOOTER #
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Created by Pengelola HPC/DGX (UG-AI-CoE)",
        br(),
        link("https://www.hpc-hub.gunadarma.ac.id/kontak/tim-pengembang-aplikasi","Universitas Gunadarma"),
    ]
    layout(*myargs)



# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    text-align: center;
    background-color: #f0f2f6;
    padding: 15px 0;
    color: var(--text-color);
}

# @media (prefers-color-scheme: light) {
#     [data-testid="stMetric"] {
#         background-color: #d3d3d3; /* Warna lebih gelap untuk tema light */
#         text-align: center;
#         padding: 15px 0;
#         color: var(--text-color);
#     }
# }

# @media (prefers-color-scheme: dark) {
#     [data-testid="stMetric"] {
#         background-color: #262730; /* Warna lebih terang untuk tema dark */
#         text-align: center;
#         padding: 15px 0;
#         color: var(--text-color);
#     }
# }

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)





# ------- Streamlit ----------- #

# CREDIT FOOTER
footer()

# HALAMAN UTAMA #
#    Pengantar Halaman
menu_sidebar = st.sidebar.selectbox('MENU',('Membuat Cluster Wajib Pajak',"Membuat Dashboard"))
if menu_sidebar == 'Membuat Cluster Wajib Pajak':
    # img_side = st.sidebar.image("https://github.com/V1ewsonic/microcred2021/blob/main/barcode%20hpc%20ug.jpeg?raw=true", use_column_width=True)
    st.title(":orange[Model Machine Learning Untuk Cluster Wajib Pajak]")
    st.header('Panduan')
    st.markdown('''1. Gunakan format file CSV \
                    \n2. Pastikan kolom yang akan anda proses bertipe data _string_ atau _object_ \
                        ''')   
    #    Upload file ke Website
    
    uploaded_file = st.file_uploader('Upload File CSV',key='uploaded_file')

    if uploaded_file is not None:
    #         Kondisi file harus CSV
        file_name = uploaded_file.name
        
        start_time = time.time()
        if is_csv(file_name):
            df = data_load(uploaded_file)
            st.write(f"Jumlah Baris data : {df.shape[0]} baris")
            st.write(f"Jumlah Kolom data : {df.shape[1]} kolom")
            st.dataframe(df)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Waktu load data : {elapsed_time} detik")


        # Eksplorasi Data Button 
        # Button pertama
        with st.container(border=True):
            st.button('Eksplorasi Data', on_click=click_button1)
            if st.session_state.button1_clicked:
            
                # Statistik Deskriptif 
                st.subheader("Statistik Deskriptif Nilai Numerik")
                deskripsi_numerik = df.describe()
                st.write(deskripsi_numerik)

                deskripsi_string = df.describe(include="O")
                st.subheader("Statistik Deskriptif Nilai Kategorik")
                st.write(deskripsi_string)

                # Cek nilai NULL
                st.subheader("Pengecekkan Nilai Null")
                cek_null = df.isnull().sum()
                st.write(cek_null)

                # Viz : Banyaknya Company Bayar pajak disuatu tahun
                st.subheader("Visualisasi Data")
                fig = ps.histogram(df, x='Sales Tax Year', color_discrete_sequence=ps.colors.qualitative.Plotly)

                max_value = df['Sales Tax Year'].value_counts().max()
                min_value = df['Sales Tax Year'].value_counts().min()
                max_year = df['Sales Tax Year'].value_counts().idxmax()
                min_year = df['Sales Tax Year'].value_counts().idxmin()

                fig.add_annotation(
                x=max_year,
                y=max_value,
                text=f'Max: {max_value}',
                showarrow=True,
                arrowhead=1,
                bgcolor="lightgreen"
                    )

                fig.add_annotation(
                x=min_year,
                y=min_value,
                text=f'Min: {min_value}',
                showarrow=True,
                arrowhead=1,
                bgcolor="lightcoral"
                    )

                fig.update_layout(
                title={
                    'text': 'Distributi Pemasukan Pajak Pertahun',
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Tahun',
                yaxis_title='Jumlah Data',
                template='plotly_dark'
                        )

                st.plotly_chart(fig,theme=None)

                ### 
                # Group by 'Selling Period' and calculate the mean of 'Taxable Sales and Purchases'
                trend_df = df.groupby('Selling Period')['Taxable Sales and Purchases'].mean().reset_index()

                # Sort the data by 'Selling Period'
                trend_df = trend_df.sort_values(by='Selling Period')

                # Create a line plot to visualize the trend
                fig = ps.line(trend_df, x='Selling Period', y='Taxable Sales and Purchases',
                                title='Trend Pemasukkan Pajak Negara Berdasarkan Bulan ',
                                labels={'Taxable Sales and Purchases': 'Average Sales (in Dollars)'})
                fig.update_layout(
                    
                xaxis_title='Periode',
                yaxis_title='Rata rata Pemasukan Pajak'
                        )

                st.plotly_chart(fig,theme=None)


                # Button kedua
                st.button('Bersihkan Data', on_click=click_button2)
                if st.session_state.button2_clicked:
                    progress_text = "Sedang memproses data. Mohon tunggu."
                    my_bar = st.progress(0, text=progress_text)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1, text=progress_text)
                    time.sleep(1)
                    my_bar.empty()
                    # menghapus data null
                    dfclean = df.dropna()
                    # Menghapus nilai mines pada "Taxable Sales and Purchases"
                    dfclean = dfclean[dfclean['Taxable Sales and Purchases'] >= 0]
                    dfclean = dfclean.reset_index()
                    st.header("Data Bersih")
                    st.write (dfclean)
                    st.write("Jumlah baris data setelah pembersihan dataset:", len(dfclean))

                # Button ketiga
                    st.button('Latih Data', on_click=click_button3)
                    if st.session_state.button3_clicked:
                        # Cleansing data
                        df = dfclean
                        df_filtered2013_2014 = df[(df['Sales Tax Year'] == "2013 - 2014") ]
                        dfmodel2013_2014 = df_filtered2013_2014[["Sales Tax Year","NAICS Industry Group","Sales Tax Quarter","Jurisdiction","Description","Taxable Sales and Purchases"]]
                        
                        df_filtered2014_2015 = df[(df['Sales Tax Year'] == "2014 - 2015") ]
                        dfmodel2014_2015 = df_filtered2014_2015[["Sales Tax Year","NAICS Industry Group","Sales Tax Quarter","Jurisdiction","Description","Taxable Sales and Purchases"]]

                        df_filtered2015_2016 = df[(df['Sales Tax Year'] == "2015 - 2016") ]
                        dfmodel2015_2016 = df_filtered2015_2016[["Sales Tax Year","NAICS Industry Group","Sales Tax Quarter","Jurisdiction","Description","Taxable Sales and Purchases"]]

                        dfutama = df[["Selling Period","Sales Tax Year","Sales Tax Quarter","NAICS Industry Group","Sales Tax Quarter","Jurisdiction","Description","Taxable Sales and Purchases"]]
                        
                        # LabelEncoder
                        le = LabelEncoder()
                        dfmodel2013_2014['Description'] = le.fit_transform(dfmodel2013_2014['Description'])
                        Description_origin2013_2014 = le.inverse_transform(dfmodel2013_2014["Description"])
                        dfmodel2013_2014['Jurisdiction'] = le.fit_transform(dfmodel2013_2014['Jurisdiction'])
                        Jurisdiction_origin2013_2014 = le.inverse_transform(dfmodel2013_2014["Jurisdiction"])

                        dfmodel2014_2015['Description'] = le.fit_transform(dfmodel2014_2015['Description'])
                        Description_origin2014_2015 = le.inverse_transform(dfmodel2014_2015["Description"])
                        dfmodel2014_2015['Jurisdiction'] = le.fit_transform(dfmodel2014_2015['Jurisdiction'])
                        Jurisdiction_origin2014_2015 = le.inverse_transform(dfmodel2014_2015["Jurisdiction"])

                        dfmodel2015_2016['Description'] = le.fit_transform(dfmodel2015_2016['Description'])
                        Description_origin2015_2016 = le.inverse_transform(dfmodel2015_2016["Description"])
                        dfmodel2015_2016['Jurisdiction'] = le.fit_transform(dfmodel2015_2016['Jurisdiction'])
                        Jurisdiction_origin2015_2016 = le.inverse_transform(dfmodel2015_2016["Jurisdiction"])

                        dfutama['Description'] = le.fit_transform(dfutama['Description'])
                        Description_originutama = le.inverse_transform(dfutama["Description"])
                        dfutama['Jurisdiction'] = le.fit_transform(dfutama['Jurisdiction'])
                        Jurisdiction_originutama = le.inverse_transform(dfutama["Jurisdiction"])

                        # Pisahkan Data Training
                        df_train2013 = dfmodel2013_2014.iloc[:,1:]
                        df_train2014 = dfmodel2014_2015.iloc[:,1:]
                        df_train2015 = dfmodel2015_2016.iloc[:,1:]
                        df_trainutama = dfutama.iloc[:,3:]


                        #Membuat clustering data
                        st.subheader("Mencari Jumlah Cluster Menggunakan Elbow method")
                        #Elbow method
                        import time
                        start_time = time.time()
                        cs = []
                        for i in range(1, 11):
                            kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
                            kmeans.fit(df_train2013)
                            cs.append(kmeans.inertia_)
                        plt.figure(figsize=(10, 5))
                        plt.plot(range(1, 11), cs)
                        plt.title('The Elbow Method')
                        plt.xlabel('Number of clusters')
                        plt.ylabel('CS')
                        st.pyplot(plt)
                        

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        st.write(f"Program execution time: {elapsed_time:.2f} seconds")

                        ## Clustering 2013_2014
                        # Choose the number of clusters (you can experiment with different values)
                        n_clusters = 3
                        # Create a KMeans object
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        # Fit the model to your data
                        kmeans.fit(df_train2013)
                        # Get the cluster labels for each data point
                        dfmodel2013_2014['cluster'] = kmeans.labels_

                        ## Clustering 2014_2015
                        # Choose the number of clusters (you can experiment with different values)
                        n_clusters = 3
                        # Create a KMeans object
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        # Fit the model to your data
                        kmeans.fit(df_train2014)
                        # Get the cluster labels for each data point
                        dfmodel2014_2015['cluster'] = kmeans.labels_

                        ## Clustering 2015_2016
                        # Choose the number of clusters (you can experiment with different values)
                        n_clusters = 3
                        # Create a KMeans object
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        # Fit the model to your data
                        kmeans.fit(df_train2015)
                        # Get the cluster labels for each data point
                        dfmodel2015_2016['cluster'] = kmeans.labels_

                        ## Clustering Utama
                        # Choose the number of clusters (you can experiment with different values)
                        n_clusters = 3
                        # Create a KMeans object
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        # Fit the model to your data
                        kmeans.fit(df_trainutama)
                        # Get the cluster labels for each data point
                        dfutama['cluster'] = kmeans.labels_


        # PLOT PERTAMA
                        dfmodel2013_2014['cluster'] = dfmodel2013_2014['cluster'].replace({0: 'Pajak Rendah', 1: 'Pajak Tinggi', 2: 'Pajak Sedang'})
                        # Visualize the clusters using a scatter plot
                        st.subheader("K-Means Clustering Pemasukan pajak Pada tahun 2013-2014")
                        fig = plt.figure(figsize=(10, 6))
                        sns.scatterplot(x='Taxable Sales and Purchases', y='Description', hue='cluster', data=dfmodel2013_2014, palette='viridis')
                        plt.title('K-Means Clustering')
                        plt.xlabel('Taxable Sales and Purchases')
                        plt.ylabel('Description')
                        st.pyplot(fig)
                        
                        dfmodel2013_2014['Description'] = Description_origin2013_2014
                        dfmodel2013_2014['Jurisdiction'] = Jurisdiction_origin2013_2014
                        col1 = dfmodel2013_2014.pop("cluster")
                        dfmodel2013_2014.insert(0, col1.name, col1)

                        cluster2013_2014 = dfmodel2013_2014["cluster"].value_counts()
                        st.subheader("Jumlah Cluster Yang Telah Dibuat")
                        st.write(cluster2013_2014)

                        cluster_2_2013_2014 = dfmodel2013_2014[dfmodel2013_2014["cluster"] == 'Pajak Tinggi']
                        st.subheader("Tampilan dataset setelah dilakukan Clustering")
                        st.write(dfmodel2013_2014.sample(1000).style.apply(highlight_column,axis=0))

                        st.subheader("Bidang yang merupakan cluster pajak yang tinggi pada tahun 2013-2014")
                        st.write(cluster_2_2013_2014["Description"].value_counts())


        # PLOT KEDUA
                        dfmodel2014_2015['cluster'] = dfmodel2014_2015['cluster'].replace({0: 'Pajak Rendah', 1: 'Pajak Tinggi', 2: 'Pajak Sedang'})
                        # Visualize the clusters using a scatter plot
                        st.subheader("K-Means Clustering Pemasukan pajak Pada tahun 2014-2015")
                        fig = plt.figure(figsize=(10, 6))
                        sns.scatterplot(x='Taxable Sales and Purchases', y='Description', hue='cluster', data=dfmodel2014_2015, palette='viridis')
                        plt.title('K-Means Clustering')
                        plt.xlabel('Taxable Sales and Purchases')
                        plt.ylabel('Description')
                        st.pyplot(fig)

                        dfmodel2014_2015['Description'] = Description_origin2014_2015
                        dfmodel2014_2015['Jurisdiction'] = Jurisdiction_origin2014_2015

                        col2 = dfmodel2014_2015.pop("cluster")
                        dfmodel2014_2015.insert(0, col2.name, col2)

                        cluster2014_2015 = dfmodel2014_2015["cluster"].value_counts()

                        st.subheader("Jumlah Cluster Yang Telah Dibuat")
                        st.write(cluster2014_2015)

                        cluster_2_2014_2015 = dfmodel2014_2015[dfmodel2014_2015["cluster"] == 'Pajak Tinggi']

                        st.subheader("Tampilan dataset setelah dilakukan Clustering")
                        st.write(dfmodel2014_2015.sample(1000).style.apply(highlight_column,axis=0))

                        st.subheader("Bidang yang merupakan cluster pajak yang tinggi pada tahun 2014-2015")
                        st.write(cluster_2_2014_2015["Description"].value_counts())
                        # Calculate value counts for 'Description' in both clusters
                        description_counts_2013_2014 = cluster_2_2013_2014["Description"].value_counts()
                        description_counts_2014_2015 = cluster_2_2014_2015["Description"].value_counts()

                        # Find the difference between the two value counts
                        difference_in_counts = description_counts_2014_2015.subtract(description_counts_2013_2014, fill_value=0)
                        st.subheader("Perubahan cluster pajak tinggi di antara tahun 2013-2014 dengan 2014-2015")
                        st.write(difference_in_counts)

        # PLOT KETIGA
                        dfmodel2015_2016['cluster'] = dfmodel2015_2016['cluster'].replace({0: 'Pajak Rendah', 1: 'Pajak Tinggi', 2: 'Pajak Sedang'})
                        # Visualize the clusters using a scatter plot
                        st.subheader("K-Means Clustering Pemasukan pajak Pada tahun 2015-2016")
                        fig = plt.figure(figsize=(10, 6))
                        sns.scatterplot(x='Taxable Sales and Purchases', y='Description', hue='cluster', data=dfmodel2015_2016, palette='viridis')
                        plt.title('K-Means Clustering')
                        plt.xlabel('Taxable Sales and Purchases')
                        plt.ylabel('Description')
                        st.pyplot(fig)

                        dfmodel2015_2016['Description'] = Description_origin2015_2016
                        dfmodel2015_2016['Jurisdiction'] = Jurisdiction_origin2015_2016

                        col3 = dfmodel2015_2016.pop("cluster")
                        dfmodel2015_2016.insert(0, col3.name, col3)

                        cluster2015_2016 = dfmodel2015_2016["cluster"].value_counts()

                        st.subheader("Jumlah Cluster Yang Telah Dibuat")
                        st.write(cluster2015_2016)

                        cluster_2_2015_2016 = dfmodel2015_2016[dfmodel2015_2016["cluster"] == 'Pajak Tinggi']
                        st.subheader("Tampilan dataset setelah dilakukan Clustering")
                        st.write(dfmodel2015_2016.sample(1000).style.apply(highlight_column,axis=0))

                        st.subheader("Bidang yang merupakan cluster pajak yang tinggi pada tahun 2015-2016")
                        st.write(cluster_2_2015_2016["Description"].value_counts())
                        description_counts_2014_2015 = cluster_2_2014_2015["Description"].value_counts()
                        description_counts_2015_2016 = cluster_2_2015_2016["Description"].value_counts()

                        # Find the difference between the two value counts
                        difference_in_counts2 = description_counts_2015_2016.subtract(description_counts_2014_2015, fill_value=0)
                        st.subheader("Perubahan cluster pajak tinggi di antara tahun 2014-2015 dengan 2015-2016")
                        st.write(difference_in_counts2)

        #Plot utama
                        dfutama['cluster'] = dfutama['cluster'].replace({0: 'Pajak Rendah', 1: 'Pajak Tinggi', 2: 'Pajak Sedang'})
                        # Visualize the clusters using a scatter plot
                        # st.write("K-Means Clustering Pemasukan pajak Pada tahun utama")
                        # fig = plt.figure(figsize=(10, 6))
                        # sns.scatterplot(x='Taxable Sales and Purchases', y='Description', hue='cluster', data=dfutama, palette='viridis')
                        # plt.title('K-Means Clustering')
                        # plt.xlabel('Taxable Sales and Purchases')
                        # plt.ylabel('Description')
                        # st.pyplot(fig)
                        
                        dfutama['Description'] = Description_originutama
                        dfutama['Jurisdiction'] = Jurisdiction_originutama
                        
                        clusterutama = dfutama["cluster"].value_counts()
                        # st.write(clusterutama)

                        # st.write(dfutama["Description"].value_counts())
                    
                    ###   
                        import pandas as pd
                        from functools import reduce
                        # Fungsi untuk menggabungkan DataFrame (misal: berdasarkan indeks)
                        def gabungkan_dataframe(df1, df2):
                            return pd.concat([df1, df2])

                        # List DataFrame
                        list_dataframe = [dfmodel2013_2014, dfmodel2014_2015, dfmodel2015_2016]

                        # Gabungkan semua DataFrame
                        dataframe_gabungan = reduce(gabungkan_dataframe, list_dataframe)
                        dataframe_gabunganclustertinggi = dataframe_gabungan[dataframe_gabungan["cluster"] == "Pajak Tinggi"]

                        
                        fig = plt.figure(figsize=(12, 6))
                        sns.countplot(x='Sales Tax Year', hue='Sales Tax Year', data=dataframe_gabunganclustertinggi)
                        plt.title('Distribusi Cluster Pajak Tinggi')
                        plt.xlabel('Periode Waktu (dalam tahun)')
                        plt.ylabel('Jumlah')
                        plt.xticks(rotation=45)

                        # Mendapatkan data untuk anotasi
                        ax = plt.gca()
                        for p in ax.patches:
                            height = p.get_height()
                            x = p.get_x() + p.get_width() / 2
                            y = p.get_y() + height + 0.05
                            ax.annotate(f'{height}', (x, y), ha='center')
                        st.pyplot(fig)

                        @st.cache_data
                        def convert_df(df):
                            return df.to_csv(index=False).encode('utf-8')

                        csv = convert_df(dfutama)

                        # Create a download button
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name='dataset.csv',
                            mime='text/csv',
                        )
                        container = st.container(border=True)
                        multi = '''Universitas Gunadarma (UG) memiliki kerjasama dengan Direktorat Jenderal Pajak (DJP) Kementerian Keuangan Republik Indonesia. Kerjasama tersebut menaungi kegiatan dengan berbagai program studi, seperti: Program Studi Akuntansi, Informatika, Sistem Informasi, Manajemen.
                              Pemanfaatan Artificial Intelligence (AI) dan teknologinya menjadi salah satu tema diskusi antara UG dan DJP. 
                              Simulasi contoh kasus data pajak mengenai cluster wajib pajak yang telah dikembangkan oleh UG menjadi ilustrasi yang dapat digunakan oleh DJP dalam rencana pengembangan sistem berbasis AI untuk Fraud Detection (didiskusikan oleh DJP dengan nara sumber dari UG dalam Focus Group Discussion Tax Crime Handling System).
                            '''
                        container.markdown("##  **Informasi terkait aplikasi simulasi:**")
                        container.markdown(multi)
                            


if menu_sidebar == 'Membuat Dashboard':
    #    Upload file ke Website
    with st.sidebar:
        st.session_state['uploaded_file'] = None
        uploaded_file_Dashboard = st.file_uploader('Upload File CSV') 
            
    #   Kondisi file harus CSV

        if uploaded_file_Dashboard is not None:
            file_name = uploaded_file_Dashboard.name
            if is_csv(file_name):
                df = data_load(uploaded_file_Dashboard)
            year_list = list(df["Sales Tax Year"].unique())[::-1]
            selected_year = st.selectbox('Select a year', year_list)
            df_selected_year = df[df["Sales Tax Year"] == selected_year]
            # df_selected_year_sorted = df_selected_year.sort_values(by="population", ascending=False)

            cluster_list = list(df_selected_year['cluster'].unique())[::-1]
            selected_cluster = st.selectbox('Select a cluster', cluster_list)
            df_selected_cluster = df_selected_year[df_selected_year["cluster"] == selected_cluster]
            grouped_by_description = df_selected_cluster.groupby('Description')['Taxable Sales and Purchases'].sum().reset_index()
            # Sort the grouped data by 'Taxable Sales and Purchases' in descending order
            grouped_by_description_sorted = grouped_by_description.sort_values(by='Taxable Sales and Purchases', ascending=False)

            # img_side = st.sidebar.image("https://github.com/V1ewsonic/microcred2021/blob/main/barcode%20hpc%20ug.jpeg?raw=true", use_column_width=True)
            # st.write("Created by Pengelola HPC/DGX (UG-AI-CoE)")
    if uploaded_file_Dashboard is not None:
        # PLOT
        # # Visualize the clusters using a scatter plot
        st.header(f":orange[Dashboard Cluster {selected_cluster} {selected_year}]")
        df_scaterplot = df_selected_year.copy()
        le = LabelEncoder()
        df_scaterplot['Description'] = le.fit_transform(df_scaterplot['Description'])
        df_scaterplot['Jurisdiction'] = le.fit_transform(df_scaterplot['Jurisdiction'])
        col = st.columns((2, 4, 2.5), gap='medium')

        with col[0]:
            st.markdown('#### Statistik Deskriptif')
            # Menghitung metrik
            avg_purchases = grouped_by_description_sorted['Taxable Sales and Purchases'].mean()
            total_purchases = grouped_by_description_sorted['Taxable Sales and Purchases'].sum()
            min_purchases = grouped_by_description_sorted['Taxable Sales and Purchases'].min()
            max_purchases = grouped_by_description_sorted['Taxable Sales and Purchases'].max()

            # Menampilkan metrik di Streamlit
            def format_currency(value):
                return f'${value * 1000000:,.2f}'
            # Convert to dollars in thousands
            avg_purchases_k = avg_purchases / 1000000
            total_purchases_k = total_purchases / 1000000
            min_purchases_k = min_purchases / 1000000
            max_purchases_k = max_purchases / 1000000

            # Display metrics
            st.metric(label="Total Setor Wajib Pajak (Juta Dollar)", value=f"{total_purchases_k:.0f} M")
            st.metric(label="Rata-rata Setor Wajib Pajak (Juta Dollar)", value=f"{avg_purchases_k:.0f} M")
            st.metric(label="Nilai Maksimal Setor Wajib Pajak (Juta Dollar)", value=f"{max_purchases_k:.0f} M")
            st.metric(label="Nilai Minimal Setor Wajib Pajak (Juta Dollar)", value=f"{min_purchases_k:.0f} M")

            max_bidang = grouped_by_description_sorted.loc[grouped_by_description_sorted['Taxable Sales and Purchases'] == max_purchases, 'Description'].values[0]
            min_bidang = grouped_by_description_sorted.loc[grouped_by_description_sorted['Taxable Sales and Purchases'] == min_purchases, 'Description'].values[0]

            st.metric(label="Bidang Setor Pajak Tertinggi", value=f"{max_bidang}")
            st.metric(label="Bidang Setor Pajak Terendah", value=f"{min_bidang}")

        with col[1]:
            st.markdown('#### Grafik Wajib Pajak ')
            # Group by 'Selling Period' and calculate the mean of 'Taxable Sales and Purchases'
            # Assuming df is your DataFrame
            trend_df = df_selected_cluster.groupby('Selling Period')['Taxable Sales and Purchases'].mean().reset_index()

            # Sort the data by 'Selling Period'
            trend_df = trend_df.sort_values(by='Selling Period')

            # Create a line plot to visualize the trend
            fig1 = ps.line(trend_df, x='Selling Period', y='Taxable Sales and Purchases',
                        labels={'Taxable Sales and Purchases': 'Average Sales (in Dollars)',"Selling Period":""})
            fig1.update_layout(
                margin=dict(l=0, r=15, t=15, b=15),
                height=350
            )
            st.plotly_chart(fig1,theme=None)
            # Group by 'description' and sum 'Taxable Sales and Purchases'
            grouped_by_description = df_selected_cluster.groupby('Description')['Taxable Sales and Purchases'].sum().reset_index()
            # Sort the grouped data by 'Taxable Sales and Purchases' in descending order
            grouped_by_description_sorted = grouped_by_description.sort_values(by='Taxable Sales and Purchases', ascending=False)
            # Display the top 10 categories with the highest total taxable sales and purchases
            top_10_categories_description = grouped_by_description_sorted.head(10)

            # Group by 'description' and sum 'Taxable Sales and Purchases'
            grouped_by_Jurisdiction= df_selected_cluster.groupby('Jurisdiction')['Taxable Sales and Purchases'].sum().reset_index()
            # Sort the grouped data by 'Taxable Sales and Purchases' in descending order
            grouped_by_Jurisdiction_sorted = grouped_by_Jurisdiction.sort_values(by='Taxable Sales and Purchases', ascending=False)
            # Display the top 10 categories with the highest total taxable sales and purchases
            top_10_categories_Jurisdiction = grouped_by_Jurisdiction_sorted.head(10)

            # Create a bar chart to visualize the top 10 categories
            fig2 = ps.bar(top_10_categories_description, x='Description', y='Taxable Sales and Purchases',
                            labels={'Taxable Sales and Purchases': 'Total Sales', 'Description' : " "},
                            color='Taxable Sales and Purchases',
                            color_continuous_scale=ps.colors.sequential.Viridis)
            fig2.update_layout(
                yaxis_tickprefix='$',
                margin=dict(l=50, r=60, t=20, b=140),

            )
            st.plotly_chart(fig2,theme=None)


        with col[2]:
            st.markdown('#### Sampel Data '+ selected_cluster )
            st.dataframe(df_selected_cluster[["Jurisdiction","NAICS Industry Group","Description" , "Taxable Sales and Purchases"]].sample(5))

            st.markdown('#### Daerah '+ selected_cluster )
            st.dataframe(
                grouped_by_Jurisdiction_sorted.head(7),
                column_order=("Jurisdiction", "Taxable Sales and Purchases"),
                hide_index=True,
                width=None,
                column_config={
                    "Jurisdiction": st.column_config.TextColumn(
                        "Daerah"
                    ),
                    "Taxable Sales and Purchases": st.column_config.NumberColumn("Setor Wajib Pajak",

                    )
                    
                }
            )

# [["Selling Period","Sales Tax Year","Sales Tax Quarter",,"Sales Tax Quarter","Jurisdiction","Description","Taxable Sales and Purchases"]]
            with st.expander('About', expanded=True):
                st.write('''
                    - Data: [DATA.NY.GOV](https://data.ny.gov/Government-Finance/Taxable-Sales-And-Purchases-Quarterly-Data-Beginni/ny73-2j3u/about_data).
                    - :orange[**Statistik Deskriptif**]: Analisis data untuk memberikan gambaran atau deskripsi yang jelas tentang data tersebut. 
                    - :orange[**Grafik Wajib Pajak**]: Grafik berdasarkan periode dan Bidang Wajib Pajak
                    - :orange[**Sampel Data Pajak Tinggi**]: Sampel Data Secara Acak Wajib Pajak
                    - :orange[**Daerah Pajak Tinggi**]: Daerah Bidang tersebut Melaporkan Pajak
                    ''')
    
 












