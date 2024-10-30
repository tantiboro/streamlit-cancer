import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly
import plotly.graph_objects as go

def get_clean_data():
    df = pd.read_csv('./data/data.csv')
    df = df.drop(['id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def plot_data(df):
    fig = go.Figure(data=go.Splom(
                  dimensions=[dict(label='Pregnancies', values=df['radius_mean']),
                              dict(label='Glucose', values=df['radius_mean']),
                              dict(label='BloodPressure', values=df['texture_mean']),
                              dict(label='SkinThickness', values=df['texture_mean']),
                              dict(label='Insulin', values=df['symmetry_mean']),
                              dict(label='BMI', values=df['symmetry_mean']),
                              dict(label='DiabPedigreeFun', values=df['smoothness_se']),
                              dict(label='Age', values=df['smoothness_se'])],
                  marker=dict(color=df['diagnosis'],
                              size=5,
                              colorscale='Bluered',
                              line=dict(width=0.5,
                                        color='rgb(230,230,230)')),
                  #text=textd,
                  diagonal=dict(visible=False)))

    title = "Scatterplot Matrix (SPLOM) for Diabetes Dataset<br>Data source:"+\
        " <a href='https://www.kaggle.com/uciml/pima-indians-diabetes-database/data'>[1]</a>"
    fig.update_layout(title=title,
                  dragmode='select',
                  width=1000,
                  height=1000,
                  hovermode='closest')

    return fig
    

def get_model():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    df = get_clean_data()

    # scale predictors and split data
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    return model, scaler


def create_radar_chart(input_data):

    input_data = get_scaled_values_dict(input_data)
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )
    
    return fig


def create_input_form(data):
    

    st.sidebar.header("Cell Nuclei Details")
    slider_labels = [("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
                     ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
                     ("Smoothness (mean)",
                      "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
                     ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)",
                                                              "concave points_mean"),
                     ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)",
                                                            "fractal_dimension_mean"),
                     ("Radius (se)", "radius_se"), ("Texture (se)",
                                                    "texture_se"), ("Perimeter (se)", "perimeter_se"),
                     ("Area (se)", "area_se"), ("Smoothness (se)", "smoothness_se"),
                     ("Compactness (se)",
                      "compactness_se"), ("Concavity (se)", "concavity_se"),
                     ("Concave points (se)",
                      "concave points_se"), ("Symmetry (se)", "symmetry_se"),
                     ("Fractal dimension (se)",
                      "fractal_dimension_se"), ("Radius (worst)", "radius_worst"),
                     ("Texture (worst)", "texture_worst"), ("Perimeter (worst)",
                                                            "perimeter_worst"),
                     ("Area (worst)", "area_worst"), ("Smoothness (worst)",
                                                      "smoothness_worst"),
                     ("Compactness (worst)",
                      "compactness_worst"), ("Concavity (worst)", "concavity_worst"),
                     ("Concave points (worst)",
                      "concave points_worst"), ("Symmetry (worst)", "symmetry_worst"),
                     ("Fractal dimension (worst)", "fractal_dimension_worst")]

    input_data = {}

    for label, col in slider_labels:
        input_data[col] = st.sidebar.slider(
            label, float(data[col].min()), float(
                data[col].max()), float(data[col].mean())
        )

    return input_data


def get_scaled_values_dict(values_dict):
    # Define a Function to Scale the Values based on the Min and Max of the Predictor in the Training Data
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in values_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def display_predictions(input_data, model, scaler):
    import streamlit as st

    import numpy as np
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)

    st.subheader('Cell cluster prediction')
    st.write("The cell cluster is: ")

    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.write("<span class='diagnosis bright-green'>Benign</span>",
                 unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis bright-red'>Malignant</span>",
                 unsafe_allow_html=True)

    st.write("Probability of being benign: ",
             model.predict_proba(input_data_scaled)[0][0])
    st.write("Probability of being malignant: ",
             model.predict_proba(input_data_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def create_app():
    import streamlit as st
    

    st.set_page_config(page_title="Breast Cancer Diagnosis",
                       page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")

    # load css
    with open("./assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)

    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    data = get_clean_data()
    input_data = create_input_form(data)
    df = get_clean_data()
    model, scaler = get_model()
    
    col1, col2, col3= st.columns([1, 1, 3])
    
    
    with col1:
        radar_chart = create_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        # load the model
        display_predictions(input_data, model, scaler)
        
    with col3:
        plolty_plot = plot_data(df)
        st.plotly_chart(plolty_plot, use_container_width=True)


def main():
    #EDA
    # df = get_clean_data()
    # plot_data(df)

    # MODEL
    # model = get_model()
    # print("Model: ", model)

    # APP
    create_app()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
