import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score
from scipy.stats import zscore
from scipy.stats.mstats import winsorize

st.set_page_config(page_title="ML Project GUI", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None

page = st.sidebar.radio("Navigation", ["1. File Upload", "2. Data Visualization", "3. Preprocessing", "4. Model Selection", "5. Model Evaluation"])

if page == "1. File Upload":
    st.header("1. File Upload")
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(st.session_state.df)

elif page == "2. Data Visualization":
    st.header("2. Data Visualization")
    if st.session_state.df is not None:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        plot_type = st.selectbox("Select Plot Type", ["Line Plot", "Scatter Plot", "Box Plot"])
        
        if plot_type == "Line Plot" and numeric_cols:
            col = st.selectbox("Select Column", numeric_cols)
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x=df.index, y=col, ax=ax)
            st.pyplot(fig)
            
        elif plot_type == "Scatter Plot" and len(numeric_cols) >= 2:
            col1 = st.selectbox("Select X axis", numeric_cols)
            col2 = st.selectbox("Select Y axis", numeric_cols)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
            st.pyplot(fig)
            
        elif plot_type == "Box Plot" and numeric_cols:
            col = st.selectbox("Select Column for Box Plot", numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Please upload a file in Page 1.")

elif page == "3. Preprocessing":
    st.header("3. Preprocessing")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("Current Dataset Shape:", df.shape)
        
        st.subheader("Drop Irrelevant Columns")
        cols_to_drop = st.multiselect("Select columns to drop", df.columns)
        if st.button("Drop Selected Columns"):
            if cols_to_drop:
                df_before = df.copy()
                df = df.drop(columns=cols_to_drop)
                st.session_state.df = df
                st.success("Columns dropped successfully!")
                st.markdown("### Dataset Before")
                st.dataframe(df_before)
                st.info(f"Operation: Dropped columns: {', '.join(cols_to_drop)}")
                st.markdown("### Dataset After")
                st.dataframe(df)
            else:
                st.warning("Please select at least one column to drop.")
                
        st.subheader("Encoding")
        encode_col = st.selectbox("Select column to encode", df.columns)
        encode_method = st.radio("Encoder", ["Label Encoder", "One-Hot Encoder"])
        if st.button("Apply Encoding"):
            df_before = df.copy()
            if encode_method == "Label Encoder":
                le = LabelEncoder()
                df[encode_col] = le.fit_transform(df[encode_col].astype(str))
            else:
                ohe = OneHotEncoder(sparse_output=False) 
                encoded_data = ohe.fit_transform(df[[encode_col]])
                encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out([encode_col]), index=df.index)
                col_idx = df.columns.get_loc(encode_col)
                df_before_parts = df.iloc[:, :col_idx]
                df_after_parts = df.iloc[:, col_idx + 1:]
                df = pd.concat([df_before_parts, encoded_df, df_after_parts], axis=1)
            st.session_state.df = df
            st.success("Encoding applied!")
            st.markdown("### Dataset Before")
            st.dataframe(df_before)
            st.info(f"Operation: {encode_method} applied to column: {encode_col}")
            st.markdown("### Dataset After")
            st.dataframe(df)
            
        st.subheader("Normalization")
        norm_method = st.radio("Scaler", ["Standard Scaler", "MinMax Scaler"])
        if st.button("Apply Scaling"):
            df_before = df.copy()
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                if norm_method == "Standard Scaler":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.session_state.df = df
                st.success("Scaling applied!")
                st.markdown("### Dataset Before")
                st.dataframe(df_before)
                st.info(f"Operation: {norm_method} applied to numeric columns")
                st.markdown("### Dataset After")
                st.dataframe(df)
            else:
                st.warning("No numeric columns found for scaling.")
            
        st.subheader("Missing Values")
        impute_method = st.selectbox("Imputer", ["Simple Imputer", "KNN Imputer", "Iterative Imputer"])
        if st.button("Apply Imputation"):
            df_before = df.copy()
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                if impute_method == "Simple Imputer":
                    imputer = SimpleImputer(strategy='mean')
                elif impute_method == "KNN Imputer":
                    imputer = KNNImputer()
                else:
                    imputer = IterativeImputer()
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                st.session_state.df = df
                st.success("Imputation applied!")
                st.markdown("### Dataset Before")
                st.dataframe(df_before)
                st.info(f"Operation: Imputation using {impute_method}")
                st.markdown("### Dataset After")
                st.dataframe(df)
            else:
                st.warning("No numeric columns found for imputation.")
            
        st.subheader("Outliers")
        outlier_col = st.selectbox("Select column for outliers", df.select_dtypes(include=np.number).columns, key="outlier_col")
        outlier_method = st.selectbox("Method", ["IQR", "Z-score", "Winsorization", "Clipping"], key="outlier_method")
        if st.button("Apply Outlier Handling"):
            df_before = df.copy()
            if outlier_method == "IQR":
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[outlier_col] < (Q1 - 1.5 * IQR)) | (df[outlier_col] > (Q3 + 1.5 * IQR)))]
            elif outlier_method == "Z-score":
                z_scores = np.abs(zscore(df[outlier_col].dropna()))
                df = df.loc[df[outlier_col].dropna().index[z_scores < 3]]
            elif outlier_method == "Winsorization":
                df[outlier_col] = winsorize(df[outlier_col], limits=[0.05, 0.05])
            elif outlier_method == "Clipping":
                lower, upper = df[outlier_col].quantile([0.05, 0.95])
                df[outlier_col] = df[outlier_col].clip(lower=lower, upper=upper)
            st.session_state.df = df
            st.success("Outlier handling applied!")
            st.markdown("### Dataset Before")
            st.dataframe(df_before)
            st.info(f"Operation: {outlier_method} applied to column: {outlier_col}")
            st.markdown("### Dataset After")
            st.dataframe(df)

        st.subheader("Transformation")
        transform_col = st.selectbox("Select column to transform", df.select_dtypes(include=np.number).columns, key="trans_col")
        transform_method = st.selectbox("Transform", ["Log Transformation", "Power Transformation (Yeo-Johnson)", "Polynomial Features"])
        if st.button("Apply Transformation"):
            df_before = df.copy()
            if transform_method == "Log Transformation":
                shift_val = df[transform_col].min()
                if shift_val < 0:
                    df[transform_col + "_log"] = np.log1p(df[transform_col] - shift_val)
                else:
                    df[transform_col + "_log"] = np.log1p(df[transform_col])
            elif transform_method == "Power Transformation (Yeo-Johnson)":
                pt = PowerTransformer(method='yeo-johnson')
                df[transform_col + "_power"] = pt.fit_transform(df[[transform_col]])
            elif transform_method == "Polynomial Features":
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_data = poly.fit_transform(df[[transform_col]])
                poly_cols = [f"{transform_col}_poly_{i}" for i in range(poly_data.shape[1])]
                poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=df.index)
                df = pd.concat([df, poly_df], axis=1)
            st.session_state.df = df
            st.success("Transformation applied!")
            st.markdown("### Dataset Before")
            st.dataframe(df_before)
            st.info(f"Operation: {transform_method} applied to column: {transform_col}")
            st.markdown("### Dataset After")
            st.dataframe(df)

        st.write("---")
        st.subheader("Target-Dependent Operations")
        st.info("Feature Selection and Handling Imbalanced Data require knowing your Target Column first.")
        target_col_prep = st.selectbox("Select Target Column for these operations", df.columns, key="target_prep")
        
        st.markdown("**Feature Selection & Dimensionality Reduction**")
        fs_method = st.selectbox("Method", ["PCA (Reduce to 2 cols)", "Recursive Feature Elimination (RFE)"])
        if st.button("Apply Selection/Reduction"):
            df_before = df.copy()
            X = df.drop(columns=[target_col_prep]).select_dtypes(include=np.number)
            y = df[target_col_prep]
            if fs_method == "PCA (Reduce to 2 cols)":
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(X)
                pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'], index=df.index)
                df = pd.concat([pca_df, y], axis=1)
                st.session_state.df = df
                st.success("PCA applied!")
            elif fs_method == "Recursive Feature Elimination (RFE)":
                model = DecisionTreeClassifier() if y.dtype == 'O' else LinearRegression()
                rfe = RFE(model, n_features_to_select=max(1, len(X.columns)//2))
                rfe.fit(X, y)
                selected_features = X.columns[rfe.support_]
                df = df[list(selected_features) + [target_col_prep]]
                st.session_state.df = df
                st.success("RFE applied!")
            st.markdown("### Dataset Before")
            st.dataframe(df_before)
            st.info(f"Operation: {fs_method} applied (Target: {target_col_prep})")
            st.markdown("### Dataset After")
            st.dataframe(df)

        st.markdown("**Handling Imbalanced Data**")
        imb_method = st.selectbox("Method", ["Oversampling (SMOTE)", "Undersampling"])
        if st.button("Apply Resampling"):
            df_before = df.copy()
            X = df.drop(columns=[target_col_prep]).select_dtypes(include=np.number)
            y = df[target_col_prep]
            try:
                if imb_method == "Oversampling (SMOTE)":
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X, y)
                else:
                    rus = RandomUnderSampler(random_state=42)
                    X_res, y_res = rus.fit_resample(X, y)
                
                df_res = pd.DataFrame(X_res, columns=X.columns)
                df_res[target_col_prep] = y_res
                df = df_res
                st.session_state.df = df
                st.success("Resampling applied!")
                st.markdown("### Dataset Before")
                st.dataframe(df_before)
                st.info(f"Operation: {imb_method} applied (Target: {target_col_prep})")
                st.markdown("### Dataset After")
                st.dataframe(df)
            except Exception as e:
                st.error("Error: This method requires the Target to be categorical (Classification), and no missing values.")

        st.write("---")
        st.markdown("### Updated Dataset Preview")
        st.dataframe(st.session_state.df)
    else:
        st.warning("Please upload a file.")

elif page == "4. Model Selection":
    st.header("4. Model Selection")
    if st.session_state.df is not None:
        df = st.session_state.df
        target_col = st.selectbox("Select Target Column", df.columns)
        all_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect("Select Features for Training", all_features, default=all_features)
        
        if not selected_features:
            st.warning("Please select at least one feature.")
        else:
            features = df[selected_features].select_dtypes(include=np.number)
            target = df[target_col]
            task_type = st.radio("Select Task", ["Classification", "Regression", "Clustering (K-Means)"])
            st.session_state.task_type = task_type
            
            if task_type == "Classification":
                model_choice = st.selectbox("Choose Algorithm", ["Decision Tree", "Logistic Regression", "SVM", "Random Forest", "K-Nearest Neighbors", "Naive Bayes", "Neural Networks"])
            elif task_type == "Regression":
                model_choice = st.selectbox("Choose Algorithm", ["Linear Regression"])
            else:
                model_choice = "K-Means"
                n_clusters = st.number_input("Number of clusters", min_value=2, value=3)

            if st.button("Train Model"):
                if target.isna().any():
                    valid_indices = target.dropna().index
                    features = features.loc[valid_indices]
                    target = target.loc[valid_indices]
                    st.info("Note: Rows with missing Target values were automatically dropped before training.")

                if features.isna().any().any():
                    st.error("Error: Your features (X) contain missing values (NaN). Please go back to Page 3 and apply 'Missing Values Imputation' first.")
                else:
                    if task_type in ["Classification", "Regression"]:
                        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        
                        if model_choice == "Decision Tree":
                            model = DecisionTreeClassifier()
                        elif model_choice == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000)
                        elif model_choice == "SVM":
                            model = SVC()
                        elif model_choice == "Random Forest":
                            model = RandomForestClassifier()
                        elif model_choice == "K-Nearest Neighbors":
                            model = KNeighborsClassifier()
                        elif model_choice == "Naive Bayes":
                            model = GaussianNB()
                        elif model_choice == "Neural Networks":
                            model = MLPClassifier(max_iter=1000)
                        elif model_choice == "Linear Regression":
                            model = LinearRegression()
                            
                        model.fit(X_train, y_train)
                        st.session_state.model = model
                        st.success(f"{model_choice} trained successfully!")
                        
                    else:
                        model = KMeans(n_clusters=n_clusters)
                        model.fit(features)
                        st.session_state.model = model
                        st.session_state.X_test = features
                        st.success("K-Means clustering applied!")
    else:
        st.warning("Please upload and preprocess a file.")

elif page == "5. Model Evaluation":
    st.header("5. Model Evaluation")
    if st.session_state.model is not None:
        model = st.session_state.model
        X_test = st.session_state.X_test
        task_type = st.session_state.task_type
        
        if task_type in ["Classification", "Regression"]:
            y_test = st.session_state.y_test
            predictions = model.predict(X_test)
            
            if task_type == "Classification":
                acc = accuracy_score(y_test, predictions)
                st.metric("Accuracy Score", round(acc, 4))
            elif task_type == "Regression":
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                st.metric("Mean Squared Error", round(mse, 4))
                st.metric("R2 Score", round(r2, 4))
        else:
            labels = model.labels_
            sil_score = silhouette_score(X_test, labels)
            st.metric("Silhouette Score", round(sil_score, 4))
    else:
        st.warning("Please train a model in Page 4 first.")