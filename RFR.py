import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Machine Learning for Everyone")

st.write("""
## **Random Forest Regression Methods**

""")

classifier_name = st.sidebar.selectbox(
    "Select a Regression Model", ("Random Forest",))

parameter_test_size = st.sidebar.slider(
    "test size (fraction)", 0.02, 0.90, 0.05)
st.sidebar.write("test size: ", parameter_test_size)
parameter_n_estimators = st.sidebar.slider("n estimators", 1, 100, 10)
st.sidebar.write("n  estimators: ", parameter_n_estimators)


st.sidebar.info("""
[More information](http://gonzalezmaw.pythonanywhere.com/)
""")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader('Dataset')
    st.write(df)
    # X = df['Temperature'].values
    # y = df['Revenue'].values
    # X = df.iloc[:, 0].values #Seleccionar los datos de la primera fila
    # y = df.iloc[:, 1].values #Seleccionar los datos de la segunda fila
    # Using all column except for the last column as X
    # Using all column except for the last column as X
    X = df.iloc[:, :-1].values
    X_name = df.iloc[:, 0].name
    y_name = df.iloc[:, 1].name
    y = df.iloc[:, -1].values  # Selecting the last column as Y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameter_test_size)
    # Fitting Random Forest Regression to the dataset

    regressor = RandomForestRegressor(
        n_estimators=parameter_n_estimators, random_state=0)
    regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

    y_pred = regressor.predict(X_test.reshape(-1, 1))
    st.subheader('Comparation')
    df = pd.DataFrame({'Real Values': y_test.reshape(-1),
                       'Predicted Values': y_pred.reshape(-1)})

    st.write(df)

    # # Visualising the Random Forest Regression Results

    st.subheader('Coefficient of determination ($R^2$):')
    resultR2 = round(r2_score(y_test, y_pred), 6)

    st.info(resultR2)

    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    # chart = st.line_chart(X_test, y_test)
    figure1, ax = plt.subplots()
    ax.scatter(X_test, y_test, label="Real values", color='green')
    ax.scatter(X_test, y_pred, label="Predicted values", color='red')
    plt.title('Random Forest Regression')
    plt.xlabel(X_name)
    plt.ylabel(y_name)
    plt.legend(loc=4)
    st.pyplot(figure1)

    figure2, ax2 = plt.subplots()
    ax2.plot(X_grid, regressor.predict(X_grid),
             label="Prediction", color='black')
    plt.title('Random Forest Regression')
    plt.xlabel(X_name)
    plt.ylabel(y_name)
    plt.legend(loc=4)
    st.pyplot(figure2)

    # st.set_option('deprecation.showPyplotGlobalUse', False)
else:
    st.info('Awaiting for CSV file to be uploaded.')
