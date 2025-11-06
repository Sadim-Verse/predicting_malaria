import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from torchvision import models
from collections import OrderedDict



# App Styling
st.set_page_config(page_title="Malaria Cell Detection", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f2f2f2;;  /* soft gray */
        color: #222;             /* dark text for contrast */
        font-family: 'Poppins', sans-serif;
    }
    img {
        border-radius: 10px;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: #111;
    }

    .prediction-box {
        padding: 10px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #e0e0e0;;
        color: #333;
        font-family: 'Poppins', sans-serif;
    }

    p {
        color: #333;
        font-family: 'Poppins', sans-serif;
    }

    .stMarkdown hr {
    border: none;
    height: 2px;
    background-color: #333;
    margin: 20px auto;
    width: 100%;          /* 👈 makes it full width */
    border-radius: 10px;
    }

    /* Target the entire file uploader container */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #aaa;
        border-radius: 12px;
        padding: 20px;
    }

    [data-testid="stFileUploader"] section {
        background-color: #a5c8ff;
        color: #333;
        border: none;
        border-radius: 15px;
        padding: 16px 16px;
        font-family: 'Poppins', sans-serif;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #0078ff;
        color: #333;
        border: none;
        border-radius: 15px;
        padding: 16px 16px;
        font-family: 'Poppins', sans-serif;
    }
    [data-testid="stFileUploader"] section button:hover {
    background-color: #d5e6ff;

    </style>
    """,
    unsafe_allow_html=True
)


# Model Loading Function
@st.cache_resource
def load_model(model_name):
    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        checkpoint = torch.load(
            r"resnet_model.pth",
            map_location=torch.device('cpu')
        )
    elif model_name == "VGG19":
        model = models.vgg19(weights=None)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
        checkpoint = torch.load(
            r"vgg19_model.pth",
            map_location=torch.device('cpu')
        )

    # If the checkpoint has the "module." prefix (from DataParallel), remove it
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "")  # remove 'module.' if present
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


# Model Selection
st.sidebar.markdown("### 🔽 Choose Model")
model_choice = st.sidebar.radio("", ["ResNet50", "VGG19"])
model = load_model(model_choice)


st.sidebar.markdown("---")


# Sidebar Information
st.sidebar.title("📖 About This Project")
st.sidebar.markdown(
    """
    ### **Malaria Cell Detection**

    Malaria remains a leading cause of illness and death in many developing regions, 
    particularly among children and pregnant women.

    While the disease is treatable if detected early, accurate diagnosis still depends 
    on manual microscopic examination of blood smear slides — a process that is:
    - Slow  
    - Subjective  
    - Resource-intensive  
    - Error-prone  

    This project develops an **AI-powered computer vision system** that can automatically 
    classify microscope images of blood smears as either **'Parasitized'** or **'Uninfected'**, 
    helping automate early malaria diagnosis.

    **System Workflow:**
    - Takes microscope images of blood smears (thin or thick films).  
    - Identifies whether the sample is infected or uninfected.  

    **Optimization Metric:**  
    - Primary: Recall  
    - Supporting: F1-score, Accuracy, Precision  

    **Model Performance:**
    - **ResNet50**
        - Recall: 0.9782 
        - Accuracy: 0.9728 
        - Precision: 0.9677  
        - F1-score: 0.9729

    - **VGG19**
        - Recall: 0.9771
        - Accuracy: 0.9728  
        - Precision: 0.9687  
        - F1-score: 0.9729  

    ---
    ⚙️ *This project demonstrates how deep learning can support medical professionals 
    by improving diagnostic accuracy, reducing workload, and saving lives.*
    """
)


# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Main App
st.title(f"🧫 Malaria Cell Detection System - *{model_choice}*")

uploaded_file = st.file_uploader("Upload a cell image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=False, width=250)
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    label = "🦠 Infected" if pred == 1 else "✅ Uninfected"
    # color = "#ff4b4b" if pred == 1 else "#3CB371"
    st.markdown(f"### Prediction: **{label}**")
    # st.markdown(
    #     f"<div class='prediction-box' style='background-color:{color}; color:white;'>{label}</div>",
    #     unsafe_allow_html=True
    # )


# Conclusion
st.markdown(
    """
    ---
    **Note:** This model is designed for research and educational purposes.  
    It should not replace professional medical diagnosis.
    """,
    unsafe_allow_html=True
)
