import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def SLR():
    parameter_test_size = st.sidebar.slider(
        "test size (fraction)", 0.02, 0.90, 0.2)
    st.sidebar.write("test size: ", parameter_test_size)

    st.sidebar.info("""
    [More information](http://gonzalezmaw.pythonanywhere.com/)
    """)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Header name
        X_name = df.iloc[:, 0].name
        y_name = df.iloc[:, 1].name

        # Using all column except for the last column as X
        X = df.iloc[:, :-1].values
        # Selecting the last column as Y
        y = df.iloc[:, -1].values

        showData = st.checkbox('Show Dataset')
        if showData:
            st.subheader('Dataset')
            st.write(df)

        # st.write("Null values: ", df.info())
            figure1, ax = plt.subplots()
        # ax.scatter(X, y, label="Dataset", color='Blue')
            ax.scatter(X, y, label="Dataset")
            plt.title('Dataset')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            plt.legend()
            st.pyplot(figure1)

        # Taking N% of the data for training and (1-N%) for testing:
        num = int(len(df)*(1-parameter_test_size))
        # training data:
        data = df
        train = df[:num]
        # Testing data:
        test = df[num:]

        st.write("Complete data: ", len(df))
        st.write("Data to train: ", len(train))
        st.write("Data to test: ", len(test))

        # Training the model:
        regr = linear_model.LinearRegression()
        train_x = np.array(train[[X_name]])
        train_y = np.array(train[[y_name]])

        regr.fit(train_x, train_y)
        coefficients = regr.coef_
        intercept = regr.intercept_

        st.subheader("""
        **Regression**
        """)
        # Slope:
        st.write("Slope:")
        st.info(coefficients[0])
        # st.info(coefficients)
        # Inercept:
        st.write("Intercept:")
        st.info(intercept)

        # Predicting values for the whole dataset
        predicted_data = regr.predict(data[[X_name]])

        # Predicting values for the whole dataset
        predicted_train = regr.predict(train[[X_name]])

        # Predicting values for testing data
        predicted_test = regr.predict(test[[X_name]])

        st.write('Coefficient of determination ($R^2$):')
        resultR2 = r2_score(y, regr.predict(data[[X_name]]))

        st.info(round(resultR2, 6))

        figure2, ax = plt.subplots()
        # ax.scatter(X, y, label="Dataset", color='Blue')
        ax.scatter(data[X_name], data[y_name], label="Dataset")
        ax.plot(data[X_name], predicted_data, label="Dataset", color="Red")
        plt.title('Complete Dataset')
        plt.xlabel(X_name)
        plt.ylabel(y_name)
        # plt.legend()
        st.pyplot(figure2)

        figure3, ax = plt.subplots()
        ax.scatter(train[X_name], train[y_name],
                   label="Dataset", color="Green")
        ax.plot(train[X_name], predicted_train, label="Dataset", color="Red")
        plt.title('Training Dataset')
        plt.xlabel(X_name)
        plt.ylabel(y_name)
        st.pyplot(figure3)

        figure4, ax = plt.subplots()
        ax.scatter(test[X_name], test[y_name], label="Dataset", color="Green")
        ax.plot(test[X_name], predicted_test, label="Dataset", color="Red")
        plt.title('Test Dataset')
        plt.xlabel(X_name)
        plt.ylabel(y_name)
        st.pyplot(figure4)

        st.subheader("Prediction:")
        X_new = st.number_input('Input a number:')

        st.write("Result:")
        y_new = np.reshape(X_new, (1, -1))
        y_new = regr.predict(y_new)
        st.info(y_new[0])

    else:
        st.info('Awaiting for CSV file to be uploaded.')

        return
