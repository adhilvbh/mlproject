import gradio as gr
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Define function to make predictions


def predict_datapoint(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=float(reading_score),
        writing_score=float(writing_score)
    )
    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return results[0]


# Define inputs
gender = gr.Dropdown(choices=["female", "male"], label="Gender")
race_ethnicity = gr.Dropdown(choices=[
    "group A", "group B", "group C", "group D", "group E"], label="Race/Ethnicity")
parental_level_of_education = gr.Dropdown(choices=["some high school", "high school", "some college",
                                                   "associate's degree", "bachelor's degree", "master's degree"], label="Parental Level of Education")
lunch = gr.Dropdown(choices=["standard", "free/reduced"], label="Lunch")
test_preparation_course = gr.Dropdown(
    choices=["none", "completed"], label="Test Preparation Course")
reading_score = gr.Slider(
    minimum=0, maximum=100, label="Reading Score")
writing_score = gr.Slider(
    minimum=0, maximum=100, label="Writing Score")

# Create interface
iface = gr.Interface(fn=predict_datapoint,
                     inputs=[gender, race_ethnicity, parental_level_of_education,
                             lunch, test_preparation_course, reading_score, writing_score],
                     outputs="text",
                     title="Student Performance Predictor")

# Launch the interface
iface.launch(share=True)
