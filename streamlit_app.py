import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import io

# Cache the model loading to ensure it's only done once
@st.cache_resource
def load_model_once():
    model = load_model('model6.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cache the encoders and scaler to ensure they're only loaded once
@st.cache_resource
def load_encoders_and_scaler():
    with open('Source_encoder.pkl', 'rb') as f:
        source_encoder = pickle.load(f)
    with open('Family_encoder.pkl', 'rb') as f:
        family_encoder = pickle.load(f)
    with open('Species_encoder.pkl', 'rb') as f:
        species_encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return source_encoder, family_encoder, species_encoder, scaler, feature_names

# Load the model and encoders/scalers once
model = load_model_once()
source_encoder, family_encoder, species_encoder, scaler, feature_names = load_encoders_and_scaler()

# Cache data loading to prevent reloading
@st.cache_data
def load_data():
    try:
        data = pd.read_excel('Cleaned_MODEL_DATA.xlsx', sheet_name='Sheet1')
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

data = load_data()

if data is not None:
    # Extract unique options only from the model's training data
    species_options = sorted(species_encoder.classes_.tolist())
    family_options = sorted(family_encoder.classes_.tolist())
    source_options = sorted(source_encoder.classes_.tolist())

    # Dynamically extract antibiotic columns based on '_I' suffix in the column name
    antibiotic_columns = [col for col in data.columns if '_I' in col]

    # Define genotype columns
    genotype_columns = ['AMPC', 'SHV', 'TEM', 'CTXM1', 'CTXM2', 'CTXM825', 'CTXM9', 'VEB', 'PER', 'GES', 'ACC', 'CMY1MOX', 'CMY11', 'DHA', 'FOX', 'ACTMIR', 'KPC', 'OXA', 'NDM', 'IMP', 'VIM', 'SPM', 'GIM']

    def preprocess_input(form_data):
        try:
            # Convert form data to DataFrame
            input_data = pd.DataFrame([form_data])

            # Encode categorical features using previously fitted label encoders
            input_data['Source'] = source_encoder.transform(input_data['Source'])
            input_data['Family'] = family_encoder.transform(input_data['Family'])
            input_data['Species'] = species_encoder.transform(input_data['Species'])

            # Map antibiotic statuses
            for col in antibiotic_columns:
                input_data[col] = input_data[col].map({'Susceptible': 0, 'Intermediate': 1, 'Resistant': 2})

            # Ensure input data has all necessary columns in the correct order
            X = input_data[feature_names]

            # Scale the input data
            X = scaler.transform(X)
            return X
        except KeyError as e:
            st.error(f"Key error during preprocessing: {e}")
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
    def generate_pdf(prediction_results, fig_bar, fig_pie):
        pdf = FPDF()
        pdf.add_page()
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stButton>button {
            color: white;
            background-color: #1f3a68;
            border-radius: 10px;
        }
        .stApp {
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='hover-section'>
            <h3 style='text-align: left;'>Input Selections:</h3>
            <p><b>Species:</b> Select the bacterial species to analyze for antibiotic resistant genes.</p>
            <p><b>Family:</b> Choose the bacterial family relevant to the analysis.</p>
            <p><b>Source:</b> Indicate the sample source for the bacterial analysis.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Streamlit app with custom title and description
    st.sidebar.image('logo.png', width=300)  # Adjust the file path and width as needed
    st.sidebar.title("Welcome to PHEGEN 🧬")
    st.sidebar.write("**Gene Prediction for Antibiotic Resistance.**")
    st.sidebar.markdown("---")  # Divider
    st.sidebar.header("Why PHEGEN?")
    st.sidebar.write(
    """
    Welcome to **PHEGEN**:
    PHEGEN is a powerful tool that uses data science and machine learning to predict genes associated with antibiotic resistance. By analyzing antibiotic profiles, it helps users identify and respond to emerging resistance threats more effectively. Users can select bacterial species, family, and sample source, and input antibiotic resistance statuses (e.g., Amikacin, Ampicillin, Ceftazidime). Based on these inputs, PHEGEN offers insights into potential gene presence, aiding in decision-making to combat antibiotic resistance and enhance public health outcomes.
    """
)
    st.sidebar.markdown("---")  # Divider
    st.sidebar.header("About")
    st.sidebar.write(
    """
    PHEGEN is a cutting-edge tool that predicts genes associated with antibiotic resistance using advanced data science and machine learning. It enables healthcare professionals and researchers to quickly identify resistance threats by analyzing bacterial species, families, sample sources, and antibiotic resistance statuses. PHEGEN provides precise insights to guide interventions and improve public health strategies against antibiotic-resistant bacteria.
    """
    )

    st.sidebar.markdown("---")  # Divider

    # Contact or feedback section
    st.sidebar.write("Created by Kashif For feedback, contact: [ks615502@gmail.com](mailto:ks615502@gmail.com)")

    st.markdown("<p style='text-align: center; color: #6c757d;'>Select the options below to predict the presence of genes based on antibiotic resistance profiles.</p>", unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        species = st.selectbox('Select Species:', species_options)
    with col2:
        family = st.selectbox('Select Family:', family_options)
    with col3:
        source = st.selectbox('Select Source:', source_options)

    # Organize antibiotic inputs in two columns
    st.subheader('Select Antibiotic Resistance Status:')
    col4, col5 = st.columns(2)

    form_data = {
        'Species': species,
        'Family': family,
        'Source': source
    }

    for i, antibiotic in enumerate(antibiotic_columns):
        if i % 2 == 0:
            with col4:
                form_data[antibiotic] = st.radio(
                    f'{antibiotic}:',
                    ('Susceptible', 'Intermediate', 'Resistant')
                )
        else:
            with col5:
                form_data[antibiotic] = st.radio(
                    f'{antibiotic}:',
                    ('Susceptible', 'Intermediate', 'Resistant')
                )

# Prediction button with a unique key
if st.button('Predict Genes', key='predict_genes_button'):
    with st.spinner('Predicting...'):
        try:
            X = preprocess_input(form_data)
            predictions = model.predict(X)

            # Prepare the output in terms of gene presence
            prediction_results = {gene: int(pred > 0.5) for gene, pred in zip(genotype_columns, predictions[0])}

            st.subheader('Prediction Results')
            st.write('Genes presence (1: Present, 0: Not Present)')
            st.table(pd.DataFrame(prediction_results.items(), columns=['Gene', 'Presence']))

            # Visualize the results with a bar chart
            st.subheader('Gene Presence Visualization')

            # Bar chart
            gene_names = list(prediction_results.keys())
            gene_presence = list(prediction_results.values())

            # Create a bar chart
            fig_bar, ax_bar = plt.subplots()
            ax_bar.barh(gene_names, gene_presence, color='skyblue')
            ax_bar.set_xlabel('Presence (1: Present, 0: Not Present)')
            ax_bar.set_title('Gene Presence Prediction Results')
            st.pyplot(fig_bar)

            # Pie chart for gene presence
            st.subheader('Gene Presence Distribution')

            # Calculate the counts of present and not present genes
            gene_count = [gene_presence.count(1), gene_presence.count(0)]
            labels = ['Present', 'Not Present']
            colors = ['#4CAF50', '#FF6347']

            # Create a pie chart
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(gene_count, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
            ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax_pie.set_title('Distribution of Gene Presence')
            st.pyplot(fig_pie)

            # Function for generating the PDF report
            def generate_pdf(prediction_results, fig_bar, fig_pie):
                pdf = FPDF()
                pdf.add_page()

                # Add title
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, "Gene Prediction Report", ln=True, align='C')

                # Add prediction results table
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "Prediction Results:", ln=True)

                pdf.set_font("Arial", size=10)
                for gene, presence in prediction_results.items():
                    pdf.cell(0, 10, f"{gene}: {'Present' if presence == 1 else 'Not Present'}", ln=True)

                # Save bar chart as image
                bar_chart_io = io.BytesIO()
                fig_bar.savefig(bar_chart_io, format='PNG')
                bar_chart_io.seek(0)
                pdf.image(bar_chart_io, x=10, y=None, w=100)

                # Save pie chart as image
                pie_chart_io = io.BytesIO()
                fig_pie.savefig(pie_chart_io, format='PNG')
                pie_chart_io.seek(0)
                pdf.image(pie_chart_io, x=110, y=None, w=100)

                # Save PDF to a BytesIO object
                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                pdf_output.seek(0)

                return pdf_output

            # Add button for downloading the report with a unique key
            if st.button('Download Report', key='download_report_button'):
                pdf_file = generate_pdf(prediction_results, fig_bar, fig_pie)
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_file,
                    file_name="gene_prediction_report.pdf",
                    mime="application/pdf"
                )

        except KeyError as e:
            st.error(f"Error in input data: {e}. Please check your input options.")
        except Exception as e:
            st.error(f"Unexpected error during prediction: {e}")
