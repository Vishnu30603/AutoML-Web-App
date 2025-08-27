import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pycaret.classification import (
    setup as setup_classification,
    compare_models as compare_models_classification,
    pull as pull_classification,
    save_model as save_model_classification,
    predict_model as predict_model_classification,
    plot_model as plot_model_classification,
    get_config as get_config_classification
)
from pycaret.regression import (
    setup as setup_regression,
    compare_models as compare_models_regression,
    pull as pull_regression,
    save_model as save_model_regression,
    predict_model as predict_model_regression,
    plot_model as plot_model_regression,
    get_config as get_config_regression
)

from pycaret.classification import interpret_model
from pycaret.regression import interpret_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering

# ---------------------- Helper Functions ----------------------

def plot_distribution(df, numerical_columns):
    for col in numerical_columns:
        st.write(f"Distribution for {col}:") 
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

def plot_categorical(df, categorical_columns):
    for col in categorical_columns:
        st.write(f"Distribution for {col}:")
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

def plot_boxplot(df, numerical_columns):
    st.write("Boxplots for Numerical Columns:")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[numerical_columns], ax=ax) 
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_correlation_matrix(df):
    st.header("Correlation Matrix")
    corr = df.select_dtypes(include=['number']).corr()  
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def save_and_display_plot(plot_func, plot_name):
    try:
        plot_func(save=True)
        if os.path.exists(f"{plot_name}.png"):
            st.image(f"{plot_name}.png")
        else:
            st.warning(f"Could not save {plot_name} plot.")
    except Exception as e:
        st.warning(f"{plot_name} not available: {e}")

def plot_clusters_pca(data, labels, n_clusters, title):
    """Visualize clusters using PCA (2D reduction)."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    df_plot = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_plot["Cluster"] = labels
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="viridis", ax=ax)
    ax.set_title(f"{title} (PCA Projection)")
    st.pyplot(fig)

def clean_dataframe(df):
    object_columns = df.select_dtypes(include=['object']).columns
    for column in object_columns:
        try:
            df[column] = df[column].astype('category')
        except Exception as e:
            st.warning(f"Could not convert column {column} to categorical: {e}")
            df[column] = df[column].apply(str)
    return df

# ---------------------- App Sidebar ----------------------
with st.sidebar:
    st.image("automl.png", use_column_width=True)
    choice = st.radio("📌 Navigation", ["Upload Data", "Explore Data", "Build Model", "Download Model"])
    st.markdown("---")
    # Quick Settings
    st.write("⚙️ Quick settings")
    random_seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1, help="Controls reproducibility.")
    st.caption("A fixed seed ensures reproducible results.")
    st.markdown("---")
    st.info("Welcome to AutoML!")

if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

# ---------------------- Upload Data ----------------------
if choice == "Upload Data":
    st.title("Upload Your Dataset for Modelling!")
    file = st.file_uploader("Upload your .csv file here")
    if file:
        try:
            df = pd.read_csv(file, index_col=None)
            df = clean_dataframe(df)
            df.to_csv("sourcedata.csv", index=None)
            st.success("File uploaded and saved successfully!")
            st.write("Data preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------------- Exploratory Data ----------------------
if choice == "Explore Data":
    st.title("Automated Exploratory Data Analysis")
    if 'df' in locals() and not df.empty:
        st.header("Dataset")
        st.dataframe(df)
        st.header("Data Overview")
        st.write("Shape of the dataset:", df.shape)
        st.write("Data Types:", df.dtypes)
        st.write("Summary Statistics:", df.describe())
        st.write("Missing Values:", df.isnull().sum())
        st.write("Unique Values:", df.nunique())
        st.header("Distribution of Numerical Columns")
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        plot_distribution(df, numerical_columns)
        plot_boxplot(df, numerical_columns)
        st.header("Distribution of Categorical Columns")
        categorical_columns = df.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            plot_categorical(df, categorical_columns)
        else:
            st.write("No categorical columns found.")
        plot_correlation_matrix(df)
    else:
        st.warning("Please upload a dataset to proceed.")

# ---------------------- Build Model ----------------------
if choice == "Build Model":
    st.title("Model Building")

    if 'df' in locals() and not df.empty:

        # ------------------- Step 1: User Inputs -------------------
        problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression", "Clustering"])
        if problem_type != "Clustering":
            target = st.selectbox("Select Your Target", df.columns)

        features = st.multiselect('Select Features to Exclude from Model', options=df.columns.tolist(), default=[])
        preprocess = st.checkbox('Preprocess Data', value=True)
        fix_imbalance = st.checkbox('Fix Imbalance', value=False)
        remove_outliers = st.checkbox('Remove Outliers', value=False)
        outliers_threshold = st.slider('Outliers Threshold', min_value=0.01, max_value=0.1, value=0.05)
        fold_count = st.slider('Number of Cross-Validation Folds', min_value=3, max_value=10, value=5)
        n_select_models = st.slider('Number of Models to Compare', min_value=1, max_value=5, value=3)

        if problem_type == "Clustering":
            cluster_method = st.selectbox("Select Clustering Method", ["KMeans", "Agglomerative Clustering"])
            n_clusters = st.slider('Select Number of Clusters', min_value=2, max_value=10, value=3)

        # ------------------- Step 2: Train Model -------------------
        if st.button('Train Model'):
            try:
                if problem_type != "Clustering":
                    train_df, test_df = train_test_split(
                        df, test_size=0.2, random_state=random_seed,
                        stratify=df[target] if problem_type == "Classification" else None
                    )
                    test_df.to_csv("testdata.csv", index=None)
                    train_df = train_df.dropna(subset=[target])
                    selected_features = [col for col in df.columns if col not in features and col != target]
                else:
                    train_df = df.dropna()
                    selected_features = [col for col in df.columns if col not in features]

                with st.spinner("Training model, please wait..."):
                    if problem_type == "Classification":
                        clf1 = setup_classification(
                            data=train_df[selected_features + [target]], target=target,
                            fix_imbalance=fix_imbalance, remove_outliers=remove_outliers, 
                            outliers_threshold=outliers_threshold, preprocess=preprocess,
                            fold=fold_count, verbose=False, session_id=random_seed
                        )
                        setup_df = pull_classification()
                        top_models = compare_models_classification(n_select=n_select_models)
                        compare_df = pull_classification()
                        save_model_classification(top_models[0], "best_model")

                    elif problem_type == "Regression":
                        clf1 = setup_regression(
                            data=train_df[selected_features + [target]], target=target,
                            remove_outliers=remove_outliers, outliers_threshold=outliers_threshold,
                            preprocess=preprocess, fold=fold_count, verbose=False, session_id=random_seed
                        )
                        setup_df = pull_regression()
                        top_models = compare_models_regression(n_select=n_select_models)
                        compare_df = pull_regression()
                        save_model_regression(top_models[0], "best_model")

                    else:  # Clustering
                        for col in train_df.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            train_df[col] = le.fit_transform(train_df[col])
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(train_df[selected_features])

                        model = KMeans(n_clusters=n_clusters, random_state=random_seed) if cluster_method == "KMeans" else AgglomerativeClustering(n_clusters=n_clusters)
                        labels = model.fit_predict(scaled_data)

                        plot_clusters_pca(scaled_data, labels, n_clusters, cluster_method)

                        if len(set(labels)) > 1:
                            score = silhouette_score(scaled_data, labels)
                            st.write(f"Silhouette Score: **{score:.3f}**")
                        else:
                            st.warning("Silhouette Score not available (only 1 cluster formed).")

                        pd.to_pickle(model, "best_cluster_model.pkl")
                        st.write("Cluster Model Summary")
                        st.dataframe(train_df[selected_features].assign(Cluster=labels))

                        st.session_state['clustering_done'] = True

                    # Save state for Classification/Regression
                    if problem_type != "Clustering":
                        st.session_state['top_models'] = top_models
                        st.session_state['problem_type'] = problem_type
                        st.session_state['setup_df'] = setup_df
                        st.session_state['compare_df'] = compare_df

            except Exception as e:
                st.error(f"Error during model training: {e}")

        # ------------------- Step 3: Post-Training Outputs -------------------
        if 'top_models' in st.session_state and st.session_state.get('problem_type') != "Clustering":
            problem_type = st.session_state['problem_type']
            setup_df = st.session_state['setup_df']
            compare_df = st.session_state['compare_df']
            top_models = st.session_state['top_models']

            st.info("Experiment Settings")
            st.dataframe(setup_df)

            st.info("Model Comparison Table")
            st.dataframe(compare_df)

            # Leaderboard Plot
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                metric_col = "F1" if (problem_type == "Classification" and "F1" in compare_df.columns) else (
                             "R2" if (problem_type == "Regression" and "R2" in compare_df.columns) else compare_df.columns[1])
                sns.barplot(data=compare_df.reset_index(), x=metric_col, y='Model', palette="viridis", ax=ax)
                ax.set_title(f"PyCaret Leaderboard ({metric_col})")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not plot leaderboard: {e}")

            # ------------------- Model Selection (No Refresh Issue) -------------------
            model_names = [f"{i+1}. {type(m).__name__}" for i, m in enumerate(top_models)]
            selected_index = st.selectbox("Select Model for Visualization", range(len(model_names)), format_func=lambda i: model_names[i])
            selected_model = top_models[selected_index]

            # Feature Importance
            st.header("Feature Importance")
            try:
                if problem_type == "Classification":
                    save_and_display_plot(lambda save: plot_model_classification(selected_model, plot='feature', save=save), "Feature Importance")
                else:
                    save_and_display_plot(lambda save: plot_model_regression(selected_model, plot='feature', save=save), "Feature Importance")
            except Exception:
                st.warning("Feature importance not available.")

            # SHAP Interpretability
            st.header("SHAP Interpretability")
            try:
                if problem_type == "Classification":
                    from pycaret.classification import interpret_model
                    interpret_model(selected_model, plot='summary', use_train_data=True)
                else:
                    from pycaret.regression import interpret_model
                    interpret_model(selected_model, plot='summary', use_train_data=True)
                fig = plt.gcf()
                fig.set_size_inches(8, 5)
                st.pyplot(fig)
                plt.clf()  
            except Exception:
                st.warning("SHAP not available or failed gracefully.")

            # Test Set Evaluation
            st.header("Test Set Evaluation")
            try:
                test_df = pd.read_csv("testdata.csv")
                metrics_list = []
                if problem_type == "Classification":
                    for model in top_models:
                        predict_model_classification(model, data=test_df)
                        metrics_list.append(pull_classification())
                else:
                    for model in top_models:
                        predict_model_regression(model, data=test_df)
                        metrics_list.append(pull_regression())
                metrics_df = pd.concat(metrics_list, keys=[f"Model {i+1}" for i in range(len(top_models))])
                st.write(metrics_df)
            except Exception as e:
                st.error(f"Error loading test set metrics: {e}")

            # Extra Plots
            if problem_type == "Classification":
                st.header("Confusion Matrix")
                save_and_display_plot(lambda save: plot_model_classification(top_models[0], plot='confusion_matrix', save=save), "Confusion Matrix")
                st.header("ROC Curve")
                save_and_display_plot(lambda save: plot_model_classification(top_models[0], plot='auc', save=save), "AUC")
            else:
                st.header("Residuals Plot")
                save_and_display_plot(lambda save: plot_model_regression(top_models[0], plot='residuals', save=save), "Residuals")

            # Pipeline
            st.header("Pipeline")
            try:
                pipe = get_config_classification('pipeline') if problem_type == "Classification" else get_config_regression('pipeline')
                st.text(pipe)
            except Exception as e:
                st.warning("Pipeline unavailable")
                st.error(e)

    else:
        st.warning("Please upload a dataset to proceed.")

# Download Model 
if choice == "Download Model":
    st.title("Download Your Model")
    download_clustering_model = st.checkbox('Download Clustering Model')
    if download_clustering_model:
        model_file = "best_cluster_model.pkl"
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                st.download_button("Download Clustering Model", f, model_file)
        else:
            st.warning("Train model first.")
    else:
        model_file = "best_model.pkl"
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                st.download_button("Download Model", f, model_file)
        else:
            st.warning("Train model first.")
    if os.path.exists("cleaned_data.csv"):
        with open("cleaned_data.csv", 'rb') as f:
            st.download_button("Download Cleaned Dataset", f, "cleaned_data.csv")
    else:
        st.warning("Cleaned dataset file not found.")
